import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset_name, dataset_results in experiment_data["ablation_type_1"].items():
    try:
        # Loss curves (Train & Validation)
        plt.figure()
        epochs = range(1, len(dataset_results["losses"]["train"]) + 1)
        plt.plot(epochs, dataset_results["losses"]["train"], label="Train Loss")
        plt.plot(epochs, dataset_results["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves - {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {dataset_name}: {e}")
        plt.close()

    try:
        # Predictions vs Ground Truth
        plt.figure()
        plt.scatter(
            dataset_results["ground_truth"], dataset_results["predictions"], alpha=0.5
        )
        plt.title(f"Predictions vs Ground Truth - {dataset_name}")
        plt.xlabel("Ground Truth HbA1c")
        plt.ylabel("Predicted HbA1c")
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating predictions vs ground truth plot for {dataset_name}: {e}"
        )
        plt.close()

    try:
        # ATT values for different learning rates
        plt.figure()
        learning_rates = [0.0005, 0.001, 0.005]
        att_values = [metric["ATT"] for metric in dataset_results["metrics"]["train"]]
        plt.bar([str(lr) for lr in learning_rates], att_values, color="skyblue")
        plt.title(f"ATT Values by Learning Rate - {dataset_name}")
        plt.xlabel("Learning Rates")
        plt.ylabel("ATT")
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_att_by_lr.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating ATT plot for {dataset_name}: {e}")
        plt.close()
