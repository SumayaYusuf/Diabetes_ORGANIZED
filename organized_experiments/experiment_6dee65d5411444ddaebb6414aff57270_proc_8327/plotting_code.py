import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    # Attempt to load the experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training and validation loss curves
    data = experiment_data["remove_hidden_layer"]["default_dataset"]["losses"]
    train_losses = data["train"]
    val_losses = data["val"]

    if train_losses and val_losses:  # Only proceed if the lists are not empty
        plt.figure()
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss", color="blue")
        plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot predictions vs ground truth (if available)
    predictions = experiment_data["remove_hidden_layer"]["default_dataset"][
        "predictions"
    ]
    ground_truth = experiment_data["remove_hidden_layer"]["default_dataset"][
        "ground_truth"
    ]

    if predictions and ground_truth:
        plt.figure()
        plt.scatter(
            ground_truth,
            predictions,
            alpha=0.6,
            edgecolors="k",
            label="Predictions vs Ground Truth",
        )
        plt.plot(
            [min(ground_truth), max(ground_truth)],
            [min(ground_truth), max(ground_truth)],
            color="red",
            label="Ideal Fit Line",
        )
        plt.title("Ground Truth vs Predictions")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
        plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()

try:
    # Plot Average Treatment Effect on the Treated (ATT) across different learning rates
    metrics = experiment_data["remove_hidden_layer"]["default_dataset"]["metrics"][
        "train"
    ]
    ATTs = [metric["ATT"] for metric in metrics if "ATT" in metric]

    if ATTs:
        plt.figure()
        plt.bar(range(len(ATTs)), ATTs, color="green", alpha=0.7)
        plt.title("ATT (Average Treatment Effect on the Treated) across Learning Rates")
        plt.xlabel("Learning Rate Index")
        plt.ylabel("ATT")
        plt.savefig(os.path.join(working_dir, "att_bar_chart.png"))
        plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
