import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # Training and Validation Losses Plot
    try:
        plt.figure()
        train_losses = experiment_data["ablation_study"]["losses"]["train"]
        val_losses = experiment_data["ablation_study"]["losses"]["val"]
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Losses")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "Train_Val_Losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Training/Validation Loss plot: {e}")
        plt.close()

    # CEE Accuracy Plot
    try:
        plt.figure()
        cee_accuracy = experiment_data["ablation_study"]["CEE_accuracy"]
        epochs = range(1, len(cee_accuracy) + 1)
        plt.plot(epochs, cee_accuracy, label="CEE Accuracy", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Causal Effect Estimation Accuracy over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "CEE_Accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CEE Accuracy plot: {e}")
        plt.close()

    # ATT (Average Treatment Effect on Treated) Plot
    try:
        plt.figure()
        att_values = experiment_data["ablation_study"]["ATT"]
        epochs = range(1, len(att_values) + 1)
        plt.plot(epochs, att_values, label="ATT", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("ATT")
        plt.title("ATT over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "ATT_Plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
else:
    print("No experiment data found or loaded!")
