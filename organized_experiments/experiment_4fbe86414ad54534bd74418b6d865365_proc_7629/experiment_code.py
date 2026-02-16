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
print(f"Using device: {device}")

# Simulate synthetic dataset
np.random.seed(42)
n_samples = 1000
age = np.random.normal(50, 15, n_samples)
exercise = np.random.choice([0, 1], size=n_samples)
cholesterol = np.random.normal(200, 30, n_samples)
medication = np.random.choice([0, 1], size=n_samples)  # Intervention
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


# Weight initialization functions
def init_weights_xavier(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)


def init_weights_kaiming(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight, nonlinearity="relu")
        nn.init.zeros_(model.bias)


# Hyperparameter tuning for weight initialization
initialization_strategies = {
    "xavier": init_weights_xavier,
    "kaiming": init_weights_kaiming,
}

experiment_data = {}

for init_name, init_func in initialization_strategies.items():
    print(f"Testing weight initialization: {init_name}")
    # Initialize model
    model = Predictor(input_dim=4).to(device)
    model.apply(init_func)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Record experiment data
    experiment_data[init_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

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
        experiment_data[init_name]["losses"]["train"].append(train_loss)

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                val_preds.append(preds.cpu().numpy())
                val_targets.append(batch["y"].cpu().numpy())
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        experiment_data[init_name]["losses"]["val"].append(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    # Store predictions and ground truth
    experiment_data[init_name]["predictions"] = np.concatenate(val_preds)
    experiment_data[init_name]["ground_truth"] = np.concatenate(val_targets)

    # Causal evaluation: Compute ATT for "Medication" intervention
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
    experiment_data[init_name]["metrics"]["train"].append({"ATT": ATT})

    print(f"Average Treatment Effect on the Treated (ATT) for {init_name}: {ATT:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
