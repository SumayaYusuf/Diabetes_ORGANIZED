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
    experiment_data = None

if experiment_data:
    for batch_size_key, data in experiment_data["hyperparam_tuning_batch_size"].items():
        try:
            # Plotting training and validation losses
            plt.figure()
            epochs = range(len(data["losses"]["train"]))
            plt.plot(epochs, data["losses"]["train"], label="Training Loss")
            plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
            plt.title(f"Loss Curves for {batch_size_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            file_name = f"{batch_size_key}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, file_name))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {batch_size_key}: {e}")
            plt.close()
