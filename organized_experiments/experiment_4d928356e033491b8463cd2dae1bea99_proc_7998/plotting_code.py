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

if experiment_data:
    for lr_key, data in experiment_data.items():
        # Plot Training/Validation Loss
        try:
            plt.figure()
            plt.plot(data["losses"]["train"], label="Training Loss")
            plt.plot(data["losses"]["val"], label="Validation Loss")
            plt.title(f"Loss Curve for {lr_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{lr_key}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {lr_key}: {e}")
            plt.close()

        # Plot ATT (If available)
        try:
            if "ATT" in data["metrics"]["train"][0]:
                ATTs = [
                    epoch_metrics["ATT"] for epoch_metrics in data["metrics"]["train"]
                ]
                plt.figure()
                plt.plot(ATTs, label="ATT")
                plt.title(f"ATT over Training for {lr_key}")
                plt.xlabel("Epochs")
                plt.ylabel("ATT")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{lr_key}_att_curve.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating ATT plot for {lr_key}: {e}")
            plt.close()
