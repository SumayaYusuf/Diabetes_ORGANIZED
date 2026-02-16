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

# Create plots
try:
    # Plot training and validation losses over epochs
    plt.figure()
    train_losses = experiment_data["hyperparam_tuning_type_1"]["dataset_1"]["losses"][
        "train"
    ]
    val_losses = experiment_data["hyperparam_tuning_type_1"]["dataset_1"]["losses"][
        "val"
    ]
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_dataset_1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Scatter plot of ground truth vs. predictions
    plt.figure()
    predictions = experiment_data["hyperparam_tuning_type_1"]["dataset_1"][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_type_1"]["dataset_1"][
        "ground_truth"
    ]
    plt.scatter(
        ground_truth[:500], predictions[:500], alpha=0.6
    )  # Plot subset to avoid clutter
    plt.xlabel("Ground Truth (HbA1c)")
    plt.ylabel("Predictions (HbA1c)")
    plt.title("Ground Truth vs. Predictions")
    plt.savefig(os.path.join(working_dir, "ground_truth_vs_predictions_dataset_1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()
