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

experiment_data = {
    "ablation_study": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulate synthetic dataset
np.random.seed(42)
n_samples = 1000
age = np.random.normal(50, 15, n_samples)
cholesterol = np.random.normal(200, 30, n_samples)
medication = np.random.choice([0, 1], size=n_samples)
exercise = np.random.choice([0, 1], size=n_samples)
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
        "Cholesterol": cholesterol,
        "Medication": medication,
        "HbA1c": hba1c,
    }
)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Convert to tensors
def prepare_tensor_data(df):
    X = df[["Age", "Cholesterol", "Medication"]].values
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


# Hyperparameter tuning: experimenting with learning rates
learning_rates = [0.0005, 0.001, 0.005]
for lr in learning_rates:
    model = Predictor(input_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        experiment_data["ablation_study"]["losses"]["train"].append(train_loss)

        model.eval()
        val_losses = []
        cee_accuracies = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())

                # Track CEE Accuracy (ground truth - simulated metric)
                avg_diff = torch.mean(torch.abs(preds - batch["y"]))
                cee_accuracies.append(1 / (1 + avg_diff.item()))

        val_loss = np.mean(val_losses)
        cee_accuracy = np.mean(cee_accuracies)
        experiment_data["ablation_study"]["losses"]["val"].append(val_loss)
        experiment_data["ablation_study"]["metrics"]["val"].append(
            {"CEE Accuracy": cee_accuracy}
        )

        print(
            f"LR {lr}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, CEE Accuracy = {cee_accuracy:.4f}"
        )

    # Causal evaluation: Compute ATT for "Medication" intervention
    model.eval()
    treated_group = train_data[train_data["Medication"] == 1]
    control_group = train_data[train_data["Medication"] == 0]

    treated_effect = model(
        torch.tensor(
            treated_group[["Age", "Cholesterol", "Medication"]].values,
            dtype=torch.float32,
        ).to(device)
    )
    control_effect = model(
        torch.tensor(
            control_group[["Age", "Cholesterol", "Medication"]].values,
            dtype=torch.float32,
        ).to(device)
    )
    ATT = (treated_effect.mean() - control_effect.mean()).item()
    experiment_data["ablation_study"]["metrics"]["train"].append({"ATT": ATT})
    print(
        f"Learning rate {lr}: Average Treatment Effect on the Treated (ATT): {ATT:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
