import yfinance as yf
import pandas as pd

def load_data(ticker, start="2015-01-01", end=None):

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df[['Close']]
        df.dropna(inplace=True)

        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        print("Error loading data:", e)
        return pd.DataFrame()