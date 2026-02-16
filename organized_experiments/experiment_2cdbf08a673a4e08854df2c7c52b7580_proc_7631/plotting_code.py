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
        # Plot training and validation losses for different configurations
        for config, results in experiment_data["layer_norm_tuning"].items():
            train_losses = results["losses"]["train"]
            val_losses = results["losses"]["val"]

            plt.figure()
            plt.plot(
                range(len(train_losses)), train_losses, label="Train Loss", color="blue"
            )
            plt.plot(
                range(len(val_losses)),
                val_losses,
                label="Validation Loss",
                color="orange",
            )
            plt.title(f"Training and Validation Losses - Layer Norm: {config}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(working_dir, f"losses_layer_norm_{config}.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    try:
        # Plot causal ATT metrics if available
        for config, results in experiment_data["layer_norm_tuning"].items():
            if "metrics" in results and results["metrics"]["train"]:
                ATT_values = [metric["ATT"] for metric in results["metrics"]["train"]]
                plt.figure()
                plt.plot(
                    range(len(ATT_values)),
                    ATT_values,
                    label="ATT (Causal Effect)",
                    color="green",
                )
                plt.title(
                    f"Average Treatment Effect on Treated (ATT) - Layer Norm: {config}"
                )
                plt.xlabel("Evaluation Points")
                plt.ylabel("ATT")
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(working_dir, f"att_layer_norm_{config}.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
