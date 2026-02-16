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

for optimizer_name, data in experiment_data.items():
    try:
        # Training and Validation losses plot
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {optimizer_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        file_path = os.path.join(working_dir, f"loss_curves_{optimizer_name}.png")
        plt.savefig(file_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {optimizer_name}: {e}")
        plt.close()

    try:
        # If ATT metrics exist, plot them
        att_values = [
            metric["ATT"] for metric in data["metrics"]["train"] if "ATT" in metric
        ]
        if att_values:
            plt.figure()
            plt.plot(att_values, label="ATT per Epoch")
            plt.title(f"ATT Evaluation for {optimizer_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Average Treatment Effect on Treated (ATT)")
            plt.legend()
            plt.grid()
            file_path = os.path.join(working_dir, f"att_plot_{optimizer_name}.png")
            plt.savefig(file_path)
            plt.close()
    except Exception as e:
        print(f"Error creating ATT plot for {optimizer_name}: {e}")
        plt.close()
