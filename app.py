import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as d
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# -------------------------------
# Streamlit App
# -------------------------------

st.title("📈 Japanese Stock Analysis App")

# Date inputs
start_date = st.date_input("Start Date", d.date(2025, 1, 1))
end_date = st.date_input("End Date", d.date(2025, 11, 15))

# Dictionary of top 10 Japanese companies
stocks = {
    "7203.T": "TOYOTA",
    "9984.T": "SOFTBANK",
    "8306.T": "MITSUBISHI_UFJ",
    "6758.T": "SONY",
    "6501.T": "HITACHI",
    "9983.T": "FAST_RETAILING",
    "7974.T": "NINTENDO",
    "8316.T": "SUMITOMO_MITSUI",
    "8035.T": "TOKYO_ELECTRON",
    "7011.T": "MITSUBISHI_HEAVY"
}

# Sidebar selection
selected_stock = st.sidebar.selectbox("Choose a company", list(stocks.values()))

# Download data
data_frames = {}
for symbol, name in stocks.items():
    df = yf.download(symbol, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)  # flatten multi-index
    df = df.reset_index()
    df['Stocks'] = name
    data_frames[name] = df

# Combine all into one DataFrame
df_all = pd.concat(data_frames.values())

# Show raw data
st.subheader("Raw Data")
st.write(df_all[df_all["Stocks"] == selected_stock].head())

# Select one stock for analysis
st.subheader(f"Analysis for {selected_stock}")
st.line_chart(df_all[df_all["Stocks"] == selected_stock].set_index("Date")["Close"])

# Stationarity Test (ADF)
st.write("### Augmented Dickey-Fuller Test")
st_data = df_all[df_all["Stocks"] == selected_stock].set_index("Date")
result = adfuller(st_data['Close'].dropna())
st.write(f"ADF Statistic: {result[0]}")
st.write(f"P-value: {result[1]}")
if result[1] < 0.05:
    st.success("✅ The series is stationary.")
else:
    st.warning("⚠️ The series is NOT stationary.")

# ARIMA Forecast
st.write("### ARIMA Forecast")
model = ARIMA(st_data['Close'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
dates = pd.date_range(start=st_data.index[-1], periods=11, freq='B')[1:]

# Plot forecast
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(st_data['Close'], label="Actual Prices")
ax.plot(dates, forecast, label="Predicted Prices", linestyle="dashed", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)
