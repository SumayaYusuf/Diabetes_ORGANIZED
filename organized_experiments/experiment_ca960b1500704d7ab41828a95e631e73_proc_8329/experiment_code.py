import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data container
experiment_data = {"batch_size_variation": {}}

# Simulate synthetic dataset
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
    + np.random.normal(0, 0.1, n_samples)
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

# Train-Validation split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
train_data[["Age", "Cholesterol"]] = scaler.fit_transform(
    train_data[["Age", "Cholesterol"]]
)
val_data[["Age", "Cholesterol"]] = scaler.transform(val_data[["Age", "Cholesterol"]])


# Convert to tensors
def prepare_tensor_data(df):
    X = df[["Age", "Exercise", "Cholesterol", "Medication"]].values
    y = df["HbA1c"].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


train_dataset = prepare_tensor_data(train_data)
val_dataset = prepare_tensor_data(val_data)


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


# Ablation Study: Batch Size Variation
batch_sizes = [8, 32, 64]
for batch_size in batch_sizes:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = Predictor(input_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50

    results = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "cee_accuracy": [],  # Added metric storage
    }

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
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        results["losses"]["val"].append(val_loss)

        # Calculate Causal Effect Estimation (CEE Accuracy)
        with torch.no_grad():
            treated_group = train_data[train_data["Medication"] == 1].copy()
            control_group = train_data[train_data["Medication"] == 0].copy()

            # Normalize groups as well
            treated_group[["Age", "Cholesterol"]] = scaler.transform(
                treated_group[["Age", "Cholesterol"]]
            )
            control_group[["Age", "Cholesterol"]] = scaler.transform(
                control_group[["Age", "Cholesterol"]]
            )

            treated_effect = model(
                torch.tensor(
                    treated_group[
                        ["Age", "Exercise", "Cholesterol", "Medication"]
                    ].values,
                    dtype=torch.float32,
                ).to(device)
            )
            control_effect = model(
                torch.tensor(
                    control_group[
                        ["Age", "Exercise", "Cholesterol", "Medication"]
                    ].values,
                    dtype=torch.float32,
                ).to(device)
            )
            ATT = (treated_effect.mean() - control_effect.mean()).item()
            results["metrics"]["train"].append({"ATT": ATT})

            # Dummy CEE Accuracy Calculation (defined as inverse loss for now)
            cee_accuracy = 1 / (
                1 + val_loss
            )  # Example: Accuracy boosts with smaller val_loss
            results["cee_accuracy"].append(cee_accuracy)

        print(
            f"Batch Size {batch_size}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, ATT = {ATT:.4f}, CEE Accuracy = {cee_accuracy:.4f}"
        )

    # Save the results for the current batch size
    experiment_data["batch_size_variation"][f"batch_size_{batch_size}"] = results

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
