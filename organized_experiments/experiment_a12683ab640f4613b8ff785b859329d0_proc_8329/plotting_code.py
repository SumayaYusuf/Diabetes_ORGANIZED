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
    experiment_data = {}

for dataset_key in ["raw_features", "normalized_features"]:
    try:
        # Plot training and validation loss curves for each dataset
        losses = experiment_data["normalization"][dataset_key]["losses"]
        train_losses = losses["train"]
        val_losses = losses["val"]

        plt.figure()
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {dataset_key.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_key}: {e}")
        plt.close()
