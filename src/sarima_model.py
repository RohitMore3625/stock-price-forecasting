from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

def train_sarima(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    joblib.dump(model_fit, "models/sarima.pkl")
    return model_fit

def forecast_sarima(model, steps=30):
    return model.forecast(steps=steps)