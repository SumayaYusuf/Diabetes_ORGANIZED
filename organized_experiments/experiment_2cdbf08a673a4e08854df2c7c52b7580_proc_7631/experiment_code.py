import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Train-test split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Convert to tensors
def prepare_tensor_data(df):
    X = df[["Age", "Exercise", "Cholesterol", "Medication"]].values
    y = df["HbA1c"].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


train_dataset = prepare_tensor_data(train_data)
val_dataset = prepare_tensor_data(val_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define models with layer normalization variations
class Predictor(nn.Module):
    def __init__(self, input_dim, norm_position=None):
        super(Predictor, self).__init__()
        self.norm_position = norm_position
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        if self.norm_position == "before":
            x = self.ln1(self.fc1(x))
            x = self.relu1(x)
            x = self.ln2(self.fc2(x))
            x = self.relu2(x)
        elif self.norm_position == "after":
            x = self.fc1(x)
            x = self.relu1(self.ln1(x))
            x = self.fc2(x)
            x = self.relu2(self.ln2(x))
        else:
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
        return self.fc3(x).squeeze()


# Model configurations
configurations = ["before", "after", None]
experiment_data = {"layer_norm_tuning": {}}

# Train and evaluate models for each configuration
for config in configurations:
    model = Predictor(input_dim=4, norm_position=config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    experiment_data["layer_norm_tuning"][str(config)] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }

    for epoch in range(n_epochs):
        # Training
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
        experiment_data["layer_norm_tuning"][str(config)]["losses"]["train"].append(
            train_loss
        )

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        experiment_data["layer_norm_tuning"][str(config)]["losses"]["val"].append(
            val_loss
        )

    # Causal evaluation
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
    experiment_data["layer_norm_tuning"][str(config)]["metrics"]["train"].append(
        {"ATT": ATT}
    )

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
