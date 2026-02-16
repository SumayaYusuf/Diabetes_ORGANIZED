import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # Plot training and validation loss curves
    try:
        train_losses = experiment_data["losses"].get("train", [])
        val_losses = experiment_data["losses"].get("val", [])
        if train_losses and val_losses:
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title("Loss Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "loss_curves.png"))
            plt.close()
        else:
            print("Train or validation losses not found in experiment data.")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot ATT if available
    try:
        att_train = [
            metric.get("ATT")
            for metric in experiment_data["metrics"]["train"]
            if "ATT" in metric
        ]
        if att_train:
            epochs = list(range(len(att_train)))
            plt.figure()
            plt.plot(epochs, att_train, label="ATT on Train Data", marker="o")
            plt.title("Average Treatment Effect on the Treated (ATT)")
            plt.xlabel("Epochs")
            plt.ylabel("ATT")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "att_curve.png"))
            plt.close()
        else:
            print("ATT metric not found in experiment data.")
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
