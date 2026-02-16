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

try:
    configs = experiment_data["hyperparam_tuning_type_1"]["hidden_layer_configs"]
    for config, results in configs.items():
        train_losses = results["losses"]["train"]
        val_losses = results["losses"]["val"]
        epochs = range(1, len(train_losses) + 1)

        try:
            plt.figure()
            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training and Validation Loss for Config {config}")
            plt.legend()
            plot_path = os.path.join(working_dir, f"loss_plot_config_{config}.png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error creating plot for config {config}: {e}")
            plt.close()
except Exception as e:
    print(f"Error processing experiment data: {e}")
