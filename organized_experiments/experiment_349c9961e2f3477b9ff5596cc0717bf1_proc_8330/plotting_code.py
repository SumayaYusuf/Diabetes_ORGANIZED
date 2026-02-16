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
    try:
        # Plot training and validation loss curves
        train_losses = experiment_data["ablation_study"]["losses"]["train"]
        val_losses = experiment_data["ablation_study"]["losses"]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Loss Curves: Training vs Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(working_dir, "loss_curves_training_validation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    try:
        # Plot validation CEE accuracy over epochs
        cee_accuracies = [
            m["CEE Accuracy"]
            for m in experiment_data["ablation_study"]["metrics"]["val"]
        ]

        plt.figure()
        plt.plot(cee_accuracies, label="CEE Accuracy", color="orange")
        plt.title("Validation CEE Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("CEE Accuracy")
        plt.grid()
        plt.savefig(os.path.join(working_dir, "validation_cee_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation CEE accuracy plot: {e}")
        plt.close()

    try:
        # Plot Average Treatment Effect on the Treated (ATT) for learning rates
        atts = [m["ATT"] for m in experiment_data["ablation_study"]["metrics"]["train"]]

        plt.figure()
        plt.plot(atts, marker="o", label="ATT")
        plt.title("Average Treatment Effect on Treated (ATT)")
        plt.xlabel("Learning Rate Index")
        plt.ylabel("ATT")
        plt.grid()
        plt.savefig(os.path.join(working_dir, "average_treatment_effect_ATT.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
