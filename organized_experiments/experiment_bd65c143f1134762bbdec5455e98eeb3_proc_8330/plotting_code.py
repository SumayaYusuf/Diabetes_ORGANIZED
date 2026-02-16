import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load saved experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    try:
        # First visualization: Loss Curves
        train_losses = experiment_data["ablation_type_1"]["no_cholesterol"]["losses"][
            "train"
        ]
        val_losses = experiment_data["ablation_type_1"]["no_cholesterol"]["losses"][
            "val"
        ]

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "loss_curves_no_cholesterol.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    try:
        # Second visualization: Predictions vs Ground Truth
        predictions = experiment_data["ablation_type_1"]["no_cholesterol"][
            "predictions"
        ]
        ground_truth = experiment_data["ablation_type_1"]["no_cholesterol"][
            "ground_truth"
        ]
        if predictions and ground_truth:
            plt.figure()
            plt.scatter(ground_truth, predictions, alpha=0.6)
            plt.title("Predictions vs Ground Truth")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.savefig(
                os.path.join(
                    working_dir, "predictions_vs_ground_truth_no_cholesterol.png"
                )
            )
            plt.close()
    except Exception as e:
        print(f"Error creating predictions vs ground truth plot: {e}")
        plt.close()

    try:
        # Third visualization: ATT values across learning rates
        att_values = experiment_data["ablation_type_1"]["no_cholesterol"]["metrics"][
            "train"
        ]
        if len(att_values) > 0:
            learning_rates = [i for i in range(len(att_values))]
            att_values = [metric["ATT"] for metric in att_values]

            plt.figure()
            plt.bar(learning_rates, att_values)
            plt.title("ATT across learning rates")
            plt.xlabel("Learning Rate Index")
            plt.ylabel("ATT")
            plt.savefig(os.path.join(working_dir, "att_values_no_cholesterol.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating ATT plot: {e}")
        plt.close()
