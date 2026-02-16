import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load the experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Define a function to save plots
def create_and_save_plot(x, y_train, y_val, plot_title, filename):
    try:
        plt.figure()
        plt.plot(x, y_train, label="Train Loss")
        plt.plot(x, y_val, label="Validation Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(working_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Error creating plot '{filename}': {e}")
        plt.close()


# Plot training and validation losses for each dropout rate
for key in experiment_data.keys():
    try:
        dropout_data = experiment_data[key]
        epochs = range(len(dropout_data["losses"]["train"]))
        train_losses = dropout_data["losses"]["train"]
        val_losses = dropout_data["losses"]["val"]

        # Sample up to every 10 epochs if there are too many points
        if len(epochs) > 50:
            sampled_indices = list(range(0, len(epochs), len(epochs) // 50))[:50]
            epochs = [epochs[i] for i in sampled_indices]
            train_losses = [train_losses[i] for i in sampled_indices]
            val_losses = [val_losses[i] for i in sampled_indices]

        plot_title = f"Loss Curves for {key.replace('_', ' ').title()}"
        filename = f"{key}_loss_curves.png"
        create_and_save_plot(epochs, train_losses, val_losses, plot_title, filename)
    except Exception as e:
        print(f"Error plotting data for {key}: {e}")
