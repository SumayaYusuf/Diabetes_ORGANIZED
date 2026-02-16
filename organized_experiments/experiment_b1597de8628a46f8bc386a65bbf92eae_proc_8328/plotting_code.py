import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot training and validation loss curves for each activation and learning rate
for act_name, act_results in experiment_data.items():
    for lr, results in act_results.items():
        try:
            # Extract losses
            train_losses = results.get("losses", {}).get("train", [])
            val_losses = results.get("losses", {}).get("val", [])

            # Plot data if it exists
            if train_losses and val_losses:
                plt.figure()
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, label="Train Loss")
                plt.plot(epochs, val_losses, label="Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title(f"Loss Curves for {act_name} Activation, LR={lr}")
                plt.legend()
                filename = f"{act_name}_lr_{lr}_loss_curves.png"
                plt.savefig(os.path.join(working_dir, filename))
                plt.close()
        except Exception as e:
            print(f"Error plotting loss curves for {act_name}, LR={lr}: {e}")
            plt.close()
