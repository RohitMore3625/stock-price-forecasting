# Fix import path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data_loader import load_data
from src.config import TICKER

# Page config
st.set_page_config(page_title="Stock Forecasting", layout="wide")

st.title("📈 Stock Forecasting - Reliance")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ARIMA", "SARIMA", "LSTM"]
)

# Load Data
df = load_data(TICKER)

if df.empty:
    st.error("❌ Failed to load stock data.")
    st.stop()

# Metric
st.metric("📊 Latest Price (₹)", round(df['Close'].iloc[-1], 2))

# ===============================
# 📈 DAILY CHART
# ===============================
st.subheader("📊 Historical Stock Prices (Daily)")

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(df.index, df['Close'])
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# ===============================
# 📅 YEARLY CHART
# ===============================
st.subheader("📅 Yearly Stock Price Trend")

yearly = df.resample('Y').mean()

fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.plot(yearly.index.year, yearly['Close'], marker='o')
plt.tight_layout()
st.pyplot(fig2, use_container_width=False)

# ===============================
# 🔮 LOAD MODEL
# ===============================
try:
    if model_choice == "ARIMA":
        model = joblib.load("models/arima.pkl")

    elif model_choice == "SARIMA":
        model = joblib.load("models/sarima.pkl")

    elif model_choice == "LSTM":
        model = load_model("models/lstm.h5")
        scaler = joblib.load("models/scaler.pkl")

except Exception as e:
    st.error(f"❌ Model load error: {e}")
    st.stop()

# ===============================
# 🔮 PREDICTION
# ===============================
st.subheader("🔮 Prediction")

if st.button("Predict Next 30 Days"):

    try:
        if model_choice in ["ARIMA", "SARIMA"]:
            pred = model.forecast(30)

            pred_df = pd.DataFrame({
                "Day": range(1, 31),
                "Predicted Price": pred
            })

        elif model_choice == "LSTM":

            # 🔥 REAL LSTM FORECAST
            data = df[['Close']].values
            scaled_data = scaler.transform(data)

            last_60 = scaled_data[-60:]
            current_input = last_60.reshape(1, 60, 1)

            lstm_preds = []

            for _ in range(30):
                next_pred = model.predict(current_input, verbose=0)
                lstm_preds.append(next_pred[0][0])

                # slide window
                current_input = np.append(current_input[:,1:,:],
                                          [[next_pred]],
                                          axis=1)

            lstm_preds = np.array(lstm_preds).reshape(-1,1)

            # inverse scaling
            lstm_preds = scaler.inverse_transform(lstm_preds)

            pred_df = pd.DataFrame({
                "Day": range(1, 31),
                "Predicted Price": lstm_preds.flatten()
            })

        # Plot prediction
        fig3, ax3 = plt.subplots(figsize=(6,3))
        ax3.plot(pred_df["Day"], pred_df["Predicted Price"], color='green')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=False)

        # Download
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("Made by Rohit 🚀")