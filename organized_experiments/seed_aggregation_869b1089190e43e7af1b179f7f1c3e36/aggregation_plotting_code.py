import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_paths = [
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_b9292d085d1b44829ec5aa81b0d3c357_proc_8328/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_6dee65d5411444ddaebb6414aff57270_proc_8327/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_6aa20a8707754940a51453f3875bd197_proc_8330/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
for path in experiment_data_paths:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")

# Aggregate and plot training/validation loss curves
try:
    train_loss_list = []
    val_loss_list = []

    for exp_data in all_experiment_data:
        data = exp_data["remove_hidden_layer"]["default_dataset"]["losses"]
        train_loss_list.append(data["train"])
        val_loss_list.append(data["val"])

    if train_loss_list and val_loss_list:  # Ensure data is available
        # Convert lists to numpy arrays for aggregation
        train_losses = np.array(train_loss_list)
        val_losses = np.array(val_loss_list)

        train_mean = np.mean(train_losses, axis=0)
        train_stderr = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        val_mean = np.mean(val_losses, axis=0)
        val_stderr = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

        epochs = range(1, len(train_mean) + 1)

        plt.figure()
        plt.plot(epochs, train_mean, label="Mean Train Loss", color="blue")
        plt.fill_between(
            epochs,
            train_mean - train_stderr,
            train_mean + train_stderr,
            color="blue",
            alpha=0.2,
            label="Train Loss Std Error",
        )
        plt.plot(epochs, val_mean, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            val_mean - val_stderr,
            val_mean + val_stderr,
            color="orange",
            alpha=0.2,
            label="Validation Loss Std Error",
        )
        plt.title("Training and Validation Loss with Std Error")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves plot: {e}")
    plt.close()

# Aggregate and plot ATT with error bars
try:
    att_list = []
    for exp_data in all_experiment_data:
        metrics = exp_data["remove_hidden_layer"]["default_dataset"]["metrics"]["train"]
        ATTs = [metric["ATT"] for metric in metrics if "ATT" in metric]
        if ATTs:
            att_list.append(ATTs)

    if att_list:  # Ensure data is available
        att_array = np.array(att_list)
        att_mean = np.mean(att_array, axis=0)
        att_stderr = np.std(att_array, axis=0) / np.sqrt(len(att_array))

        plt.figure()
        plt.bar(
            range(len(att_mean)),
            att_mean,
            yerr=att_stderr,
            color="green",
            alpha=0.7,
            capsize=5,
            label="ATT with Std Error",
        )
        plt.title("Aggregated ATT (Average Treatment Effect on the Treated)")
        plt.xlabel("Learning Rate Index")
        plt.ylabel("ATT")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_att_bar_chart.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated ATT plot: {e}")
    plt.close()
