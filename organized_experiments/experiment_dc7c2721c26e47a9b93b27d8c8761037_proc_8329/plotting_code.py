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
    exit()

# Plot training and validation losses for each noise level
for noise_level, data in experiment_data.items():
    try:
        plt.figure()
        epochs = range(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for Noise Level {noise_level}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_noise_{noise_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for noise level {noise_level}: {e}")
        plt.close()

# Plot ATT values at different noise levels
try:
    plt.figure()
    noise_levels = [key.split("_")[1] for key in experiment_data.keys()]
    att_values = [
        data["metrics"]["train"][0]["ATT"] for data in experiment_data.values()
    ]
    plt.bar(noise_levels, att_values, color="skyblue")
    plt.xlabel("Noise Level")
    plt.ylabel("ATT")
    plt.title("ATT at Different Noise Levels")
    plt.savefig(os.path.join(working_dir, "att_vs_noise_level.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
