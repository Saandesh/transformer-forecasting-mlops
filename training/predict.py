import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from models.transformer_model import TransformerTimeSeries

# Load test data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load model
model = TransformerTimeSeries()

model.load_state_dict(
    torch.load("models/transformer_model.pth")
)

model.eval()

# Predictions
with torch.no_grad():
    predictions = model(X_test_tensor)

predictions = predictions.numpy()

# Reload original data for inverse scaling
data = pd.read_csv("data/apple_stock.csv")

close_prices = pd.to_numeric(
    data['Close'],
    errors='coerce'
).values.reshape(-1,1)

scaler = MinMaxScaler()

scaler.fit(close_prices)

# Convert back to original prices
predictions_actual = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test)

# Plot
plt.figure(figsize=(14,7))

plt.plot(
    y_test_actual,
    label="Actual Prices"
)

plt.plot(
    predictions_actual,
    label="Predicted Prices"
)

plt.title("Transformer Stock Price Prediction")

plt.xlabel("Time")

plt.ylabel("Stock Price")

plt.legend()

plt.show()
