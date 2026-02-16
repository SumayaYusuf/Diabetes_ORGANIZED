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
        # Plot training and validation losses for different weight decays
        plt.figure()
        dataset_info = experiment_data["hyperparam_tuning_type_1"]["dataset_1"]
        losses = dataset_info["losses"]

        for weight_idx, weight_decay in enumerate([0, 1e-5, 1e-4, 1e-3, 1e-2]):
            if weight_idx < len(losses["train"]):  # Index-safe check
                plt.plot(
                    range(len(losses["train"][weight_idx])),
                    losses["train"][weight_idx],
                    label=f"Train Loss W={weight_decay}",
                )
                plt.plot(
                    range(len(losses["val"][weight_idx])),
                    losses["val"][weight_idx],
                    label=f"Val Loss W={weight_decay}",
                )

        plt.title("Training vs Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "training_validation_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    try:
        # Plot predictions vs ground truth for one weight_decay example
        plt.figure()
        predictions = (
            dataset_info["predictions"][0] if dataset_info["predictions"] else []
        )
        ground_truth = dataset_info["ground_truth"]

        if predictions:
            plt.scatter(
                ground_truth, predictions, alpha=0.5, label="Predicted vs Actual"
            )
            plt.plot(
                [min(ground_truth), max(ground_truth)],
                [min(ground_truth), max(ground_truth)],
                "r--",
                label="Ideal Line",
            )
            plt.title("Validation Predictions vs Ground Truth")
            plt.xlabel("Ground Truth HbA1c")
            plt.ylabel("Predicted HbA1c")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot: {e}")
        plt.close()
