from fastapi import FastAPI
import joblib

app = FastAPI()

arima = joblib.load("models/arima.pkl")

@app.get("/")
def home():
    return {"message": "Stock Prediction API"}

@app.get("/predict")
def predict():
    pred = arima.forecast(5)
    return {"prediction": pred.tolist()}