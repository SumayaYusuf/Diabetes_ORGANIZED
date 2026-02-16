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


# Define a flexible feedforward neural network
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(Predictor, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x).squeeze()


# Hyperparameter tuning configurations
hidden_layer_configs = [(64, 32), (128, 64), (32, 16), (256, 128)]
n_epochs = 50
lr = 0.001
input_dim = 4

experiment_data = {"hyperparam_tuning_type_1": {"hidden_layer_configs": {}}}

# Train and evaluate models for each config
for config in hidden_layer_configs:
    model = Predictor(input_dim=input_dim, hidden_sizes=config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    results = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }

    for epoch in range(n_epochs):
        # Training phase
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

        # Validation phase
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

        print(
            f"Config {config}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    # Save results for this hidden layer configuration
    experiment_data["hyperparam_tuning_type_1"]["hidden_layer_configs"][
        str(config)
    ] = results

# Save the experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
