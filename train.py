from src.config import *
from src.data_loader import load_data
from src.preprocessing import scale_data, create_sequences
from src.feature_engineering import add_features
from src.metrics import evaluate
from src.arima_model import train_arima
from src.sarima_model import train_sarima
from src.lstm_model import train_lstm

import mlflow
import numpy as np

# Load Data
df = load_data(TICKER)

# Feature Engineering
df = add_features(df)

# ARIMA
arima_model = train_arima(df['Close'], ARIMA_ORDER)
arima_pred = arima_model.forecast(steps=30)

# SARIMA
sarima_model = train_sarima(df['Close'], SARIMA_ORDER, SEASONAL_ORDER)
sarima_pred = sarima_model.forecast(steps=30)

# LSTM
scaled_data, scaler = scale_data(df[['Close']])
X, y = create_sequences(scaled_data, TIME_STEP)
X = X.reshape(X.shape[0], X.shape[1], 1)

lstm_model = train_lstm(X, y, EPOCHS, BATCH_SIZE)
lstm_pred = lstm_model.predict(X[-30:])

# Dummy evaluation (example)
y_true = y[-30:].flatten()
lstm_pred = lstm_pred.flatten()

mae, rmse, mape = evaluate(y_true, lstm_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

# MLflow Logging
mlflow.set_experiment("Stock Forecasting")

with mlflow.start_run():
    mlflow.log_param("model", "Hybrid")
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)

print("Training Complete ✅")


import joblib
joblib.dump(scaler, "models/scaler.pkl")