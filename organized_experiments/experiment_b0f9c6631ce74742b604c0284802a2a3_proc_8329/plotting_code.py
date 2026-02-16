import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

try:
    # Training and Validation Loss Curves
    plt.figure()
    train_losses = experiment_data["Feature_Removal_Exercise"]["losses"]["train"]
    val_losses = experiment_data["Feature_Removal_Exercise"]["losses"]["val"]
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    # CEE Accuracy Over Epochs
    plt.figure()
    cee_accuracies = [
        val_metric["CEE_Accuracy"]
        for val_metric in experiment_data["Feature_Removal_Exercise"]["metrics"]["val"]
    ]
    plt.plot(
        range(len(cee_accuracies)), cee_accuracies, label="CEE Accuracy", color="green"
    )
    plt.title("CEE Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CEE Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "cee_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CEE Accuracy plot: {e}")
    plt.close()

try:
    # Predictions vs Ground Truth Scatter Plot
    plt.figure()
    preds = experiment_data["Feature_Removal_Exercise"]["predictions"][-1]
    ground_truth = experiment_data["Feature_Removal_Exercise"]["ground_truth"][-1]
    plt.scatter(ground_truth, preds, alpha=0.6, label="Predictions vs Ground Truth")
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
        linestyle="--",
        label="Ideal",
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs Ground Truth plot: {e}")
    plt.close()

try:
    # ATT Values Bar Plot
    plt.figure()
    train_metrics = experiment_data["Feature_Removal_Exercise"]["metrics"]["train"]
    att_values = [metric["ATT"] for metric in train_metrics]
    plt.bar(range(len(att_values)), att_values, color="skyblue")
    plt.title("ATT Values for Learning Rates")
    plt.xlabel("Learning Rate Index")
    plt.ylabel("ATT")
    plt.savefig(os.path.join(working_dir, "att_values.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
