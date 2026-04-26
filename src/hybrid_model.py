import numpy as np

def hybrid_forecast(arima_pred, lstm_pred):
    return (arima_pred + lstm_pred) / 2