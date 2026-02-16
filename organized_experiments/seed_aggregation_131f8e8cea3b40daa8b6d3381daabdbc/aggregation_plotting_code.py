import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_6181f814cd0e47bfada36cfa7a9bbcf6_proc_7472/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_0e0d4fbdcdd2464b97835a449ac09647_proc_7470/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_ed005dad9c7649a5a59167fe45b0b665_proc_7471/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data from {experiment_data_path}: {e}")

if all_experiment_data:
    # Aggregate Training and Validation Losses
    try:
        train_losses_list = []
        val_losses_list = []
        for experiment_data in all_experiment_data:
            train_losses_list.append(experiment_data["losses"].get("train", []))
            val_losses_list.append(experiment_data["losses"].get("val", []))

        train_losses = np.array(
            [
                np.pad(
                    loss,
                    (0, max(len(max(train_losses_list, key=len)) - len(loss), 0)),
                    constant_values=np.nan,
                )
                for loss in train_losses_list
            ]
        )
        val_losses = np.array(
            [
                np.pad(
                    loss,
                    (0, max(len(max(val_losses_list, key=len)) - len(loss), 0)),
                    constant_values=np.nan,
                )
                for loss in val_losses_list
            ]
        )

        mean_train_loss = np.nanmean(train_losses, axis=0)
        std_train_loss = np.nanstd(train_losses, axis=0)
        mean_val_loss = np.nanmean(val_losses, axis=0)
        std_val_loss = np.nanstd(val_losses, axis=0)

        epochs = np.arange(len(mean_train_loss))

        plt.figure()
        plt.plot(epochs, mean_train_loss, label="Mean Train Loss", color="blue")
        plt.fill_between(
            epochs,
            mean_train_loss - std_train_loss,
            mean_train_loss + std_train_loss,
            color="blue",
            alpha=0.2,
        )
        plt.plot(epochs, mean_val_loss, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            mean_val_loss - std_val_loss,
            mean_val_loss + std_val_loss,
            color="orange",
            alpha=0.2,
        )
        plt.title("Aggregated Loss Curves with Standard Error")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # Aggregate ATT Metrics
    try:
        att_train_list = []
        for experiment_data in all_experiment_data:
            att_train = [
                metric.get("ATT", np.nan)
                for metric in experiment_data["metrics"]["train"]
            ]
            att_train_list.append(att_train)

        max_epochs = max(len(metrics) for metrics in att_train_list)
        att_train_data = np.array(
            [
                np.pad(metrics, (0, max_epochs - len(metrics)), constant_values=np.nan)
                for metrics in att_train_list
            ]
        )

        mean_att_train = np.nanmean(att_train_data, axis=0)
        std_att_train = np.nanstd(att_train_data, axis=0)

        epochs = np.arange(len(mean_att_train))

        plt.figure()
        plt.plot(
            epochs,
            mean_att_train,
            label="Mean ATT on Train Data",
            marker="o",
            color="green",
        )
        plt.fill_between(
            epochs,
            mean_att_train - std_att_train,
            mean_att_train + std_att_train,
            color="green",
            alpha=0.2,
        )
        plt.title("Aggregated ATT Metric with Standard Error")
        plt.xlabel("Epochs")
        plt.ylabel("ATT")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_att_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated ATT plot: {e}")
        plt.close()
