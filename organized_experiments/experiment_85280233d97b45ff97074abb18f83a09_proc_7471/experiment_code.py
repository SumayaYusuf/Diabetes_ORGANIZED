import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic data generation
np.random.seed(0)
n = 1000  # Number of samples
glucose = np.random.normal(120, 15, n)  # Random glucose levels
cholesterol = np.random.normal(200, 30, n)  # Random cholesterol
blood_pressure = np.random.normal(120, 10, n)  # Random blood pressure
intervention = (np.random.rand(n) < 0.5).astype(int)  # Random intervention (binary 0/1)
# Treatment effect: HbA1c reduction of 0.5 for treated
hbA1c = (
    7
    + (glucose / 200)
    + (cholesterol / 300)
    + (blood_pressure / 150)
    - 0.5 * intervention
    + np.random.normal(0, 0.5, n)
)

# Dataset
data = np.vstack((glucose, cholesterol, blood_pressure, intervention, hbA1c)).T
columns = ["glucose", "cholesterol", "blood_pressure", "intervention", "hbA1c"]
dataset_name = "synthetic_diabetes"

# Split data
X = data[:, :3]  # Features: glucose, cholesterol, blood pressure
T = data[:, 3]  # Treatment: intervention
Y = data[:, 4]  # Outcome: HbA1c
X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)

# Propensity score estimation using logistic regression
propensity_model = LogisticRegression()
propensity_model.fit(X_train, T_train)
propensity_scores = propensity_model.predict_proba(X_val)[:, 1]

# ATT calculation
treated_idx = T_val == 1
control_idx = T_val == 0

att_numerator = np.mean(Y_val[treated_idx]) - np.mean(Y_val[control_idx])
att = att_numerator  # Simplified ATT estimation for linear setting

# Save metrics and results
experiment_data = {
    dataset_name: {
        "metrics": {"ATT": att},
        "synth_data": {
            "glucose": glucose,
            "cholesterol": cholesterol,
            "blood_pressure": blood_pressure,
            "intervention": intervention,
            "hbA1c": hbA1c,
        },
    },
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print ATT
print(f"Average Treatment Effect on the Treated (ATT): {att}")
