from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from models.transformer_model import TransformerTimeSeries

app = FastAPI()

# Load stock data
data = pd.read_csv("data/apple_stock.csv")

close_prices = pd.to_numeric(
    data['Close'],
    errors='coerce'
).values.reshape(-1,1)

# Scale data
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

@app.get("/")

def home():

    return {
        "message": "Transformer Stock Forecast API Running"
    }

@app.get("/predict")

def predict():

    sequence_length = 30

    last_sequence = scaled_data[-sequence_length:]

    input_tensor = torch.tensor(
        last_sequence.reshape(1, sequence_length, 1),
        dtype=torch.float32
    )

    with torch.no_grad():

        prediction = model(input_tensor)

    predicted_price = scaler.inverse_transform(
        np.array(prediction.item()).reshape(-1,1)
    )

    return {
        "predicted_stock_price":
        float(predicted_price[0][0])
    }

