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
        # Plot training and validation loss curves for activation functions
        for act_func, data in experiment_data.get(
            "activation_function_tuning", {}
        ).items():
            train_losses = data["losses"]["train"]
            val_losses = data["losses"]["val"]

            plt.figure()
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title(f"Loss Curves for {act_func}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"loss_curves_{act_func}.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        plt.close()

    try:
        # Plot ATT comparison bar chart
        atts = {
            act_func: data["metrics"]["train"][-1]["ATT"]
            for act_func, data in experiment_data["activation_function_tuning"].items()
            if "ATT" in data["metrics"]["train"][-1]
        }

        if atts:
            plt.figure()
            plt.bar(atts.keys(), atts.values())
            plt.title("ATT Comparison Across Activation Functions")
            plt.xlabel("Activation Function")
            plt.ylabel("ATT")
            plt.savefig(os.path.join(working_dir, "ATT_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting ATT comparison: {e}")
        plt.close()
