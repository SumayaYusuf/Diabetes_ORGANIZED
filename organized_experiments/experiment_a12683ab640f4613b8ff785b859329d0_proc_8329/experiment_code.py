import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

experiment_data = {
    "normalization": {
        "raw_features": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "normalized_features": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
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
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Preprocessing functions
def prepare_tensor_data(df, scaler=None):
    X = df[["Age", "Exercise", "Cholesterol", "Medication"]].values
    y = df["HbA1c"].values
    if scaler:
        X = scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


# Normalization setup
scaler = StandardScaler().fit(
    train_data[["Age", "Exercise", "Cholesterol", "Medication"]]
)
normalized_train_dataset = prepare_tensor_data(train_data, scaler)
normalized_val_dataset = prepare_tensor_data(val_data, scaler)
raw_train_dataset = prepare_tensor_data(train_data)
raw_val_dataset = prepare_tensor_data(val_data)


# Loaders
def get_dataloaders(dataset):
    train_loader = DataLoader(dataset[0], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[1], batch_size=32)
    return train_loader, val_loader


datasets = {
    "raw_features": (raw_train_dataset, raw_val_dataset),
    "normalized_features": (normalized_train_dataset, normalized_val_dataset),
}


# Model definition
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


# Training loop
def train_and_evaluate(lr, train_loader, val_loader, dataset_key):
    model = Predictor(input_dim=4).to(device)
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
        experiment_data["normalization"][dataset_key]["losses"]["train"].append(
            train_loss
        )

        model.eval()
        val_losses = []
        predictions, ground_truth = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
                preds = model(batch["X"])
                loss = criterion(preds, batch["y"])
                val_losses.append(loss.item())
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(batch["y"].cpu().numpy())
        val_loss = np.mean(val_losses)
        experiment_data["normalization"][dataset_key]["losses"]["val"].append(val_loss)
        experiment_data["normalization"][dataset_key]["predictions"] = predictions
        experiment_data["normalization"][dataset_key]["ground_truth"] = ground_truth

        print(
            f"{dataset_key}, LR {lr}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )


# Run experiments
learning_rate = 0.001
for dataset_key, dataset in datasets.items():
    train_loader, val_loader = get_dataloaders(dataset)
    train_and_evaluate(learning_rate, train_loader, val_loader, dataset_key)

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
