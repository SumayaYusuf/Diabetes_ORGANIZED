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

try:
    # Iterate over learning rates to plot training and validation loss for each
    for lr, results in experiment_data["hidden_layer_width_reduction"].items():
        train_losses = results["losses"]["train"]
        val_losses = results["losses"]["val"]

        # Plot training and validation losses
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training vs Validation Loss (Learning Rate: {lr})")
        plt.legend()
        save_path = os.path.join(working_dir, f"loss_plot_lr_{lr}.png")
        plt.savefig(save_path)
        print(f"Saved plot for learning rate {lr} to {save_path}")
        plt.close()

except Exception as e:
    print(f"Error creating plots: {e}")
    plt.close()
