import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Training and validation loss curves
try:
    train_losses = experiment_data["ablation_type_1"]["mae_vs_mse"]["losses"]["train"][
        0
    ]
    val_losses = experiment_data["ablation_type_1"]["mae_vs_mse"]["losses"]["val"][0]

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves: Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "loss_curves_train_val.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training vs validation loss plot: {e}")
    plt.close()

# Predictions vs Ground Truths scatter plot
try:
    predictions = experiment_data["ablation_type_1"]["mae_vs_mse"]["predictions"]
    ground_truth = experiment_data["ablation_type_1"]["mae_vs_mse"]["ground_truth"]

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        "r--",
    )
    plt.title("Predictions vs Ground Truths")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.grid()
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
