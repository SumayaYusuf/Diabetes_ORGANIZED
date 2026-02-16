import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set up a working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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
    {"Age": age, "Cholesterol": cholesterol, "Medication": medication, "HbA1c": hba1c}
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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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


# Initialize experiment data
experiment_data = {
    "Feature_Removal_Exercise": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

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
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        experiment_data["Feature_Removal_Exercise"]["losses"]["train"].append(
            train_loss
        )

        model.eval()
        val_losses = []
        val_preds = []
        val_truths = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item())
                val_preds.extend(preds.cpu().numpy())
                val_truths.extend(batch_y.cpu().numpy())
        val_loss = np.mean(val_losses)
        experiment_data["Feature_Removal_Exercise"]["losses"]["val"].append(val_loss)
        experiment_data["Feature_Removal_Exercise"]["predictions"].append(val_preds)
        experiment_data["Feature_Removal_Exercise"]["ground_truth"].append(val_truths)

        # Compute CEE Accuracy (placeholder for real causal evaluation; adapt as required)
        cee_accuracy = 100 - abs(val_loss * 100)  # Example formula
        experiment_data["Feature_Removal_Exercise"]["metrics"]["val"].append(
            {"CEE_Accuracy": cee_accuracy}
        )

        print(
            f"Epoch {epoch + 1}, LR {lr}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, CEE Accuracy = {cee_accuracy:.2f}"
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
    experiment_data["Feature_Removal_Exercise"]["metrics"]["train"].append({"ATT": ATT})
    print(
        f"Learning rate {lr}: Average Treatment Effect on the Treated (ATT): {ATT:.4f}"
    )

# Save the results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
