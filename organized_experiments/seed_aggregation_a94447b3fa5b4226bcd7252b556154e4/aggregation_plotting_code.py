import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# List of experiment data paths
experiment_data_path_list = [
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_4d928356e033491b8463cd2dae1bea99_proc_7998/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_fffec6c300e44301b9ebafdbaf9929dc_proc_7997/experiment_data.npy",
    "experiments/2026-02-15_19-31-15_causal_discovery_intervention_attempt_0/logs/0-run/experiment_results/experiment_3bc831b944f341c68c45622f79f679ac_proc_8000/experiment_data.npy",
]

all_experiment_data = []
try:
    for exp_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if all_experiment_data:
    # Gather data across experiments
    aggregated_data = {}
    for data in all_experiment_data:
        for lr_key, lr_data in data.items():
            if lr_key not in aggregated_data:
                aggregated_data[lr_key] = {
                    "train_losses": [],
                    "val_losses": [],
                    "ATTs": [],
                }
            aggregated_data[lr_key]["train_losses"].append(lr_data["losses"]["train"])
            aggregated_data[lr_key]["val_losses"].append(lr_data["losses"]["val"])
            if "ATT" in lr_data["metrics"]["train"][0]:
                aggregated_data[lr_key]["ATTs"].append(
                    [
                        epoch_metrics["ATT"]
                        for epoch_metrics in lr_data["metrics"]["train"]
                    ]
                )

    # Aggregate results and plot
    for lr_key, results in aggregated_data.items():
        # Training and Validation Loss Aggregated Plot
        try:
            train_losses = np.array(results["train_losses"])
            val_losses = np.array(results["val_losses"])
            mean_train = np.mean(train_losses, axis=0)
            std_err_train = np.std(train_losses, axis=0) / np.sqrt(
                train_losses.shape[0]
            )
            mean_val = np.mean(val_losses, axis=0)
            std_err_val = np.std(val_losses, axis=0) / np.sqrt(val_losses.shape[0])

            plt.figure()
            epochs = np.arange(1, len(mean_train) + 1)
            plt.plot(epochs, mean_train, label="Mean Training Loss")
            plt.fill_between(
                epochs,
                mean_train - std_err_train,
                mean_train + std_err_train,
                alpha=0.3,
                label="Std Err Training Loss",
            )
            plt.plot(epochs, mean_val, label="Mean Validation Loss")
            plt.fill_between(
                epochs,
                mean_val - std_err_val,
                mean_val + std_err_val,
                alpha=0.3,
                label="Std Err Validation Loss",
            )
            plt.title(f"Aggregated Loss Curves for {lr_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{lr_key}_aggregated_loss_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curve for {lr_key}: {e}")
            plt.close()

        # ATT Aggregated Plot (if available)
        try:
            if results["ATTs"]:
                ATTs = np.array(results["ATTs"])
                mean_ATT = np.mean(ATTs, axis=0)
                std_err_ATT = np.std(ATTs, axis=0) / np.sqrt(ATTs.shape[0])

                plt.figure()
                epochs = np.arange(1, len(mean_ATT) + 1)
                plt.plot(epochs, mean_ATT, label="Mean ATT")
                plt.fill_between(
                    epochs,
                    mean_ATT - std_err_ATT,
                    mean_ATT + std_err_ATT,
                    alpha=0.3,
                    label="Std Err ATT",
                )
                plt.title(f"Aggregated ATT Curve for {lr_key}")
                plt.xlabel("Epochs")
                plt.ylabel("ATT")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{lr_key}_aggregated_ATT_curve.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated ATT plot for {lr_key}: {e}")
            plt.close()
