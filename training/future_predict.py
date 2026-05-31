import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from models.transformer_model import TransformerTimeSeries

# Load stock data
data = pd.read_csv("data/apple_stock.csv")

close_prices = pd.to_numeric(
    data['Close'],
    errors='coerce'
).values.reshape(-1,1)

# Scaling
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    close_prices
)

# Load model
model = TransformerTimeSeries()

model.load_state_dict(
    torch.load("models/transformer_model.pth")
)

model.eval()

# Last 30 days
sequence_length = 30

last_sequence = scaled_data[-sequence_length:]

future_predictions = []

current_sequence = last_sequence.copy()

future_days = 30

for _ in range(future_days):

    input_tensor = torch.tensor(
        current_sequence.reshape(1, sequence_length, 1),
        dtype=torch.float32
    )

    with torch.no_grad():

        predicted = model(input_tensor)

    predicted_value = predicted.item()

    future_predictions.append(
        predicted_value
    )

    current_sequence = np.append(
        current_sequence[1:],
        [[predicted_value]],
        axis=0
    )

# Convert back to actual prices
future_predictions_actual = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

# Plot
plt.figure(figsize=(14,7))

plt.plot(
    future_predictions_actual,
    marker='o'
)

plt.title(
    "Future 30-Day Stock Forecast"
)

plt.xlabel("Future Days")

plt.ylabel("Predicted Stock Price")

plt.grid(True)

plt.show()

