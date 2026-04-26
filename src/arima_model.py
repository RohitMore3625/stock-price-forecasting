from statsmodels.tsa.arima.model import ARIMA
import joblib

def train_arima(data, order=(5,1,0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    joblib.dump(model_fit, "models/arima.pkl")
    return model_fit

def forecast_arima(model, steps=30):
    return model.forecast(steps=steps)