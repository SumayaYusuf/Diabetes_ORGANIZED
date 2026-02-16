import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_file = os.path.join(working_dir, "experiment_data.npy")

try:
    experiment_data = np.load(experiment_file, allow_pickle=True).item()
    experiment = experiment_data["synthetic_diabetes"]
    synth_data = experiment["synth_data"]
    metrics = experiment["metrics"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

try:
    # Distribution plot for glucose
    plt.figure()
    plt.hist(
        synth_data["glucose"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    plt.title("Glucose Level Distribution")
    plt.xlabel("Glucose Level")
    plt.ylabel("Frequency")
    plt.savefig(
        os.path.join(working_dir, "synthetic_diabetes_glucose_distribution.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating glucose distribution plot: {e}")
    plt.close()

try:
    # Box plot comparing HbA1c levels for treated and control groups
    plt.figure()
    hbA1c_treated = synth_data["hbA1c"][synth_data["intervention"] == 1]
    hbA1c_control = synth_data["hbA1c"][synth_data["intervention"] == 0]
    plt.boxplot([hbA1c_treated, hbA1c_control], labels=["Treated", "Control"])
    plt.title("HbA1c Level Comparison")
    plt.ylabel("HbA1c")
    plt.savefig(os.path.join(working_dir, "synthetic_diabetes_hba1c_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HbA1c comparison plot: {e}")
    plt.close()

try:
    # Bar plot for ATT (Average Treatment Effect on the Treated)
    plt.figure()
    plt.bar(["ATT"], [metrics["ATT"]], color="cornflowerblue")
    plt.title("Average Treatment Effect on the Treated (ATT)")
    plt.ylabel("Effect Size")
    plt.savefig(os.path.join(working_dir, "synthetic_diabetes_ATT.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ATT plot: {e}")
    plt.close()
