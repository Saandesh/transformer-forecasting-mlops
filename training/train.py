import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models.transformer_model import TransformerTimeSeries

# Load data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Dataset & Loader
dataset = TensorDataset(X_train, y_train)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Model
model = TransformerTimeSeries()

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 20

mlflow.set_experiment(
    "Transformer Stock Forecasting"
)

with mlflow.start_run():

    mlflow.log_param(
        "epochs",
        epochs
    )

    mlflow.log_param(
        "learning_rate",
        0.001
    )

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for X_batch, y_batch in loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

mlflow.log_metric(
    "loss",
    epoch_loss,
    step=epoch
)

# Save model
torch.save(
    model.state_dict(),
    "models/transformer_model.pth"
)

print("Model saved successfully!")

mlflow.pytorch.log_model(
    model,
    "transformer-model"
)
