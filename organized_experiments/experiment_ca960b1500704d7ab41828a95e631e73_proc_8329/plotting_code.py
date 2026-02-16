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
    # Plot training and validation loss curves for different batch sizes
    for batch_size, results in experiment_data["batch_size_variation"].items():
        # Training and validation losses
        train_losses = results["losses"]["train"]
        val_losses = results["losses"]["val"]

        # Generate loss curve
        plt.figure()
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {batch_size}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{batch_size}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    # Plot CEE accuracy over epochs for each batch size
    for batch_size, results in experiment_data["batch_size_variation"].items():
        cee_accuracy = results["cee_accuracy"]

        # Generate accuracy curve
        plt.figure()
        epochs = range(1, len(cee_accuracy) + 1)
        plt.plot(epochs, cee_accuracy, label="CEE Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"CEE Accuracy for {batch_size}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"cee_accuracy_{batch_size}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating CEE accuracy plot: {e}")
    plt.close()
