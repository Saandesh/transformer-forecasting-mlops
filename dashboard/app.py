import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from models.transformer_model import TransformerTimeSeries

st.title("Transformer Stock Forecast Dashboard")

st.write("AI-powered stock price forecasting using Transformers")

st.divider()

stock = st.selectbox(
    "Select Stock",
    ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
)

forecast_days = st.slider(
    "Forecast Horizon (Days)",
    1,
    30,
    7
)

generate = st.button(
    "Generate Forecast"
)

# Load data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensor
X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

# Load model
model = TransformerTimeSeries()

model.load_state_dict(
    torch.load("models/transformer_model.pth")
)

model.eval()

# Generate predictions
with torch.no_grad():

    predictions = model(X_test_tensor)

predictions = predictions.numpy()

# Load original stock data
data = pd.read_csv("data/apple_stock.csv")

close_prices = pd.to_numeric(
    data['Close'],
    errors='coerce'
).values.reshape(-1,1)

# Scaling
scaler = MinMaxScaler()

scaler.fit(close_prices)

# Convert back to actual prices
predictions_actual = scaler.inverse_transform(
    predictions
)

y_test_actual = scaler.inverse_transform(
    y_test
)

current_price = float(y_test_actual[-1])

predicted_price = float(
    predictions_actual[-1]
)

change_percent = (
    (predicted_price - current_price)
    / current_price
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${current_price:.2f}"
)

col2.metric(
    "Predicted Price",
    f"${predicted_price:.2f}"
)

col3.metric(
    "Change %",
    f"{change_percent:.2f}%"
)

# Plot graph
fig, ax = plt.subplots(figsize=(14,7))

ax.plot(
    y_test_actual,
    linewidth=2,
    label="Actual Prices"
)

ax.plot(
    predictions_actual,
    linewidth=2,
    linestyle="--",
    label="Predicted Prices"
)

ax.set_title(
    "Actual vs Predicted Stock Prices"
)

ax.set_xlabel("Time")

ax.set_ylabel("Stock Price")

ax.legend()

# Show graph in dashboard
st.pyplot(fig)

st.success("Prediction generated successfully!")

