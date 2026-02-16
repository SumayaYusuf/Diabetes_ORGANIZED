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

# Plot training and validation losses over epochs (if available)
try:
    if "losses" in experiment_data["min_max_scaling"]:
        train_losses = experiment_data["min_max_scaling"]["losses"]["train"]
        val_losses = experiment_data["min_max_scaling"]["losses"]["val"]

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "train_val_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot ATT values calculated for each learning rate (if available)
try:
    if (
        "metrics" in experiment_data["min_max_scaling"]
        and experiment_data["min_max_scaling"]["metrics"]["train"]
    ):
        atts = [
            entry["ATT"]
            for entry in experiment_data["min_max_scaling"]["metrics"]["train"]
        ]
        learning_rates = [0.0005, 0.001, 0.005]

        plt.figure()
        plt.bar([str(lr) for lr in learning_rates], atts, color="skyblue")
        plt.xlabel("Learning Rate")
        plt.ylabel("ATT (Average Treatment Effect on the Treated)")
        plt.title("ATT Values for Different Learning Rates")
        plt.savefig(os.path.join(working_dir, "att_values_lr.png"))
        plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
