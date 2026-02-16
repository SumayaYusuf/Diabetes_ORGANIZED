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


# Define a configurable feedforward neural network
class Predictor(nn.Module):
    def __init__(self, input_dim, activation_fn):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            activation_fn(),
            nn.Linear(64, 32),
            activation_fn(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x).squeeze()


# Define the experiment data structure
experiment_data = {"activation_function_tuning": {}}

# List of activation functions to test
activation_functions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "Tanh": nn.Tanh,
}

# Run experiments for each activation function
for act_name, act_fn in activation_functions.items():
    print(f"Testing activation function: {act_name}")
    model = Predictor(input_dim=4, activation_fn=act_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 50
    results = {"metrics": {"train": [], "val": []}, "losses": {"train": [], "val": []}}

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

        print(
            f"Activation {act_name} - Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    # Calculate ATT for intervention
    model.eval()
    with torch.no_grad():
        X_train = train_data[["Age", "Exercise", "Cholesterol", "Medication"]].values
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

        treated_X = X_train.clone()
        treated_X[:, 3] = 1  # Set "Medication" to 1 for treated group
        control_X = X_train.clone()
        control_X[:, 3] = 0  # Set "Medication" to 0 for control group

        treated_effect = model(treated_X)
        control_effect = model(control_X)

    ATT = (treated_effect.mean() - control_effect.mean()).item()
    results["metrics"]["train"].append({"ATT": ATT})

    print(f"{act_name} - Average Treatment Effect on the Treated (ATT): {ATT:.4f}")
    experiment_data["activation_function_tuning"][act_name] = results

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
