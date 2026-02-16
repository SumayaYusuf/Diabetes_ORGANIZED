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
        epoch_settings = [50, 100, 200]
        for i, num_epochs in enumerate(epoch_settings):
            train_losses = experiment_data["hyperparam_tuning_type_1"][
                "synthetic_dataset"
            ]["losses"]["train"][i]
            val_losses = experiment_data["hyperparam_tuning_type_1"][
                "synthetic_dataset"
            ]["losses"]["val"][i]

            plt.figure()
            plt.plot(
                range(1, len(train_losses) + 1),
                train_losses,
                label="Training Loss",
                color="blue",
            )
            plt.plot(
                range(1, len(val_losses) + 1),
                val_losses,
                label="Validation Loss",
                color="orange",
            )
            plt.title(f"Loss Curves (Epochs={num_epochs})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"loss_curves_epochs_{num_epochs}.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    try:
        # Plot ATT results across epoch settings
        att_results = [
            item["ATT"]
            for item in experiment_data["hyperparam_tuning_type_1"][
                "synthetic_dataset"
            ]["metrics"]["train"]
        ]

        plt.figure()
        plt.plot(epoch_settings, att_results, marker="o", linestyle="-", color="green")
        plt.title("Average Treatment Effect on the Treated (ATT)")
        plt.xlabel("Epochs")
        plt.ylabel("ATT")
        plt.grid(True)
        plt.savefig(os.path.join(working_dir, "att_results.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
