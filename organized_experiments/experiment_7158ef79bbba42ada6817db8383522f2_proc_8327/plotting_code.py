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

# Training and validation losses over epochs
try:
    for lr_key, data in experiment_data["dropout_regularization"].items():
        plt.figure()
        epochs = range(1, 1 + len(data["losses"]["train"]))
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.title(f"Training & Validation Losses - {lr_key}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{lr_key}_loss_plot.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ATT visualization
try:
    plt.figure()
    lr_keys = list(experiment_data["dropout_regularization"].keys())
    att_values = [
        data["metrics"]["train"][-1]["ATT"]
        for data in experiment_data["dropout_regularization"].values()
    ]
    plt.bar(lr_keys, att_values)
    plt.title("Average Treatment Effect on Treated (ATT) by Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("ATT")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(working_dir, "att_bar_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
