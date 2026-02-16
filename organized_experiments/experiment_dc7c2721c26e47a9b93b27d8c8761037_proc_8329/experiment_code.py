import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

experiment_data = {}

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to generate synthetic data with varying noise levels
def generate_synthetic_data(noise_std):
    np.random.seed(42)
    n_samples = 1000
    age = np.random.normal(50, 15, n_samples)
    exercise = np.random.choice([0, 1], size=n_samples)
    cholesterol = np.random.normal(200, 30, n_samples)
    medication = np.random.choice([0, 1], size=n_samples)
    hba1c = (
        6
        + 0.02 * age
        - 0.5 * exercise
        + 0.03 * cholesterol
        - 0.7 * medication
        + np.random.normal(0, noise_std, n_samples)
    )
    data = pd.DataFrame(
        {
            "Age": age,
            "Exercise": exercise,
            "Cholesterol": cholesterol,
            "Medication": medication,
            "HbA1c": hba1c,
        }
    )
    return data


# Convert to tensors
def prepare_tensor_data(df):
    X = df[["Age", "Exercise", "Cholesterol", "Medication"]].values
    y = df["HbA1c"].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


# Define a simple feedforward neural network
class Predictor(nn.Module):
    def __init__(self, input_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x).squeeze()


# Train the model and compute ATT
def run_experiment(data, noise_level):
    results = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = prepare_tensor_data(train_data)
    val_dataset = prepare_tensor_data(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = Predictor(input_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
            optimizer.zero_grad()
            preds = model(batch["X"])
            loss = criterion(preds, batch["y"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        results["losses"]["train"].append(train_loss)

        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch["y"].cpu().numpy())
        val_loss = np.mean(val_losses)
        results["losses"]["val"].append(val_loss)
        results["predictions"] = val_preds
        results["ground_truth"] = val_true

    # Compute ATT
    model.eval()
    treated_group = train_data[train_data["Medication"] == 1]
    control_group = train_data[train_data["Medication"] == 0]

    treated_effect = model(
        torch.tensor(
            treated_group[["Age", "Exercise", "Cholesterol", "Medication"]].values,
            dtype=torch.float32,
        ).to(device)
    )
    control_effect = model(
        torch.tensor(
            control_group[["Age", "Exercise", "Cholesterol", "Medication"]].values,
            dtype=torch.float32,
        ).to(device)
    )
    ATT = (treated_effect.mean() - control_effect.mean()).item()
    results["metrics"]["train"].append({"ATT": ATT})
    print(f"Noise level {noise_level}: ATT: {ATT:.4f}")
    return results


# Different noise levels for ablation
noise_levels = [0.05, 0.1, 0.2, 0.5]
for noise in noise_levels:
    data = generate_synthetic_data(noise_std=noise)
    experiment_data[f"noise_{noise}"] = run_experiment(data, noise)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
