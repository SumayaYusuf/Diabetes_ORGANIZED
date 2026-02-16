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

for init_name, data in experiment_data.items():
    # Plot training and validation loss curves
    try:
        plt.figure()
        epochs = range(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Training Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{init_name} Initialization: Training & Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{init_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {init_name}: {e}")
        plt.close()

    # Plot predictions vs ground truth
    try:
        plt.figure()
        predictions = data["predictions"]
        ground_truth = data["ground_truth"]
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.plot(
            ground_truth, ground_truth, color="red", linestyle="--", label="Ideal Fit"
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title(f"{init_name} Initialization: Predictions vs Ground Truth")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{init_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions vs ground truth plot for {init_name}: {e}")
        plt.close()
