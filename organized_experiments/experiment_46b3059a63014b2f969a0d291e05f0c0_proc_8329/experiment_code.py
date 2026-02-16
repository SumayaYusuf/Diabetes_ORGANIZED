import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

experiment_data = {"ablation_type_1": {}}
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_synthetic_data(seed, coeffs):
    np.random.seed(seed)
    n_samples = 1000
    age = np.random.normal(50, 15, n_samples)
    exercise = np.random.choice([0, 1], size=n_samples)
    cholesterol = np.random.normal(200, 30, n_samples)
    medication = np.random.choice([0, 1], size=n_samples)
    hba1c = (
        coeffs["base"]
        + coeffs["age"] * age
        - coeffs["exercise"] * exercise
        + coeffs["cholesterol"] * cholesterol
        - coeffs["medication"] * medication
        + np.random.normal(0, 0.1, n_samples)
    )
    return pd.DataFrame(
        {
            "Age": age,
            "Exercise": exercise,
            "Cholesterol": cholesterol,
            "Medication": medication,
            "HbA1c": hba1c,
        }
    )


datasets = {
    "dataset_1": {
        "seed": 42,
        "coeffs": {
            "base": 6,
            "age": 0.02,
            "exercise": 0.5,
            "cholesterol": 0.03,
            "medication": 0.7,
        },
    },
    "dataset_2": {
        "seed": 43,
        "coeffs": {
            "base": 5.5,
            "age": 0.03,
            "exercise": 0.4,
            "cholesterol": 0.02,
            "medication": 0.6,
        },
    },
    "dataset_3": {
        "seed": 44,
        "coeffs": {
            "base": 6.5,
            "age": 0.01,
            "exercise": 0.6,
            "cholesterol": 0.04,
            "medication": 0.5,
        },
    },
    "dataset_4": {
        "seed": 45,
        "coeffs": {
            "base": 6.2,
            "age": 0.015,
            "exercise": 0.45,
            "cholesterol": 0.025,
            "medication": 0.65,
        },
    },
}


def prepare_tensor_data(df):
    X = df[["Age", "Exercise", "Cholesterol", "Medication"]].values
    y = df["HbA1c"].values
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )


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


learning_rates = [0.0005, 0.001, 0.005]

for dataset_name, config in datasets.items():
    data = generate_synthetic_data(config["seed"], config["coeffs"])
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=config["seed"]
    )
    train_dataset = prepare_tensor_data(train_data)
    val_dataset = prepare_tensor_data(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    dataset_results = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for lr in learning_rates:
        model = Predictor(input_dim=4).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        n_epochs = 50

        for epoch in range(n_epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                X, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            dataset_results["losses"]["train"].append(train_loss)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    X, y = batch[0].to(device), batch[1].to(device)
                    preds = model(X)
                    val_losses.append(criterion(preds, y).item())
            val_loss = np.mean(val_losses)
            dataset_results["losses"]["val"].append(val_loss)

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
        dataset_results["metrics"]["train"].append({"ATT": ATT})

    dataset_results["predictions"] = (
        model(
            torch.tensor(
                val_data[["Age", "Exercise", "Cholesterol", "Medication"]].values,
                dtype=torch.float32,
            ).to(device)
        )
        .cpu()
        .detach()
        .numpy()
        .tolist()
    )
    dataset_results["ground_truth"] = val_data["HbA1c"].values.tolist()
    experiment_data["ablation_type_1"][dataset_name] = dataset_results

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
