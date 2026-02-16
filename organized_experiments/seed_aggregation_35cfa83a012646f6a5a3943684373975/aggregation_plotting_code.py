import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_paths = [
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_9c469f2ff3f24ea8b56eacf9271c4d1c_proc_7631/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_d465ec34dc7441f9b729649342a680a6_proc_7632/experiment_data.npy",
]

all_experiment_data = []
try:
    for path in experiment_data_paths:
        if os.path.exists(path):
            experiment_data = np.load(path, allow_pickle=True).item()
            all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if all_experiment_data:
    aggregated_metrics = {"train_loss": [], "val_loss": [], "att": []}
    max_epochs = 0

    # Process each dataset and aggregate metrics
    for exp_data in all_experiment_data:
        for lr_key, data in exp_data.items():
            train_loss = data["losses"]["train"] if "train" in data["losses"] else []
            val_loss = data["losses"]["val"] if "val" in data["losses"] else []
            train_att = (
                [epoch_metrics["ATT"] for epoch_metrics in data["metrics"]["train"]]
                if "ATT" in data["metrics"]["train"][0]
                else []
            )

            aggregated_metrics["train_loss"].append(train_loss)
            aggregated_metrics["val_loss"].append(val_loss)
            aggregated_metrics["att"].append(train_att)
            max_epochs = max(max_epochs, len(train_loss))

    def compute_mean_and_se(data, max_epochs):
        """Helper function to compute mean and standard error"""
        aligned_data = [
            np.pad(d, (0, max_epochs - len(d)), "constant", constant_values=np.nan)
            for d in data
        ]
        aligned_array = np.array(aligned_data)
        means = np.nanmean(aligned_array, axis=0)
        ses = np.nanstd(aligned_array, axis=0) / np.sqrt(aligned_array.shape[0])
        return means, ses

    # Compute aggregated statistics
    train_loss_mean, train_loss_se = compute_mean_and_se(
        aggregated_metrics["train_loss"], max_epochs
    )
    val_loss_mean, val_loss_se = compute_mean_and_se(
        aggregated_metrics["val_loss"], max_epochs
    )
    att_mean, att_se = compute_mean_and_se(aggregated_metrics["att"], max_epochs)

    # Plot aggregated metrics
    try:
        plt.figure()
        epochs = np.arange(max_epochs)
        plt.plot(epochs, train_loss_mean, label="Train Loss Mean", color="blue")
        plt.fill_between(
            epochs,
            train_loss_mean - train_loss_se,
            train_loss_mean + train_loss_se,
            color="blue",
            alpha=0.2,
            label="Train Loss SE",
        )
        plt.plot(epochs, val_loss_mean, label="Val Loss Mean", color="orange")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_se,
            val_loss_mean + val_loss_se,
            color="orange",
            alpha=0.2,
            label="Val Loss SE",
        )
        plt.title("Aggregated Training and Validation Loss with Error Bars")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated loss: {e}")

    try:
        plt.figure()
        plt.plot(epochs, att_mean, label="ATT Mean", color="green")
        plt.fill_between(
            epochs,
            att_mean - att_se,
            att_mean + att_se,
            color="green",
            alpha=0.2,
            label="ATT SE",
        )
        plt.title("Aggregated ATT with Error Bars")
        plt.xlabel("Epochs")
        plt.ylabel("ATT")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_att_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated ATT: {e}")
