import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = "579b464db66ec23bdd0000017ce9d9b9e36c456f41f78d04d7c1e97d"
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

st.set_page_config(page_title="AI Mandi Predictor", layout="wide")
st.title("🌾 AI-Powered Mandi Price Forecast Tool")

# ---------------------------
# GET MANDIS FUNCTION
# ---------------------------
@st.cache_data
def get_mandis(state):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[state]": state
    }
    try:
        response = requests.get(URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data.get('records', []))
        if df.empty:
            return []
        return sorted(df['market'].dropna().unique())
    except Exception as e:
        st.error(f"Error fetching mandis: {str(e)}")
        return []

# ---------------------------
# FETCH DATA FUNCTION
# ---------------------------
@st.cache_data
def load_data(crop, state, mandi):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[commodity]": crop,
        "filters[state]": state,
        "filters[market]": mandi
    }
    try:
        response = requests.get(URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data.get('records', []))
        if df.empty:
            return df
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df = df.dropna(subset=['modal_price'])
        df = df.sort_values('arrival_date')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# ---------------------------
# FORECAST FUNCTION (Polynomial Regression)
# ---------------------------
def forecast_prices(df, days=7):
    df['date_numeric'] = (df['arrival_date'] - df['arrival_date'].min()).dt.days
    X = df['date_numeric'].values.reshape(-1, 1)
    y = df['modal_price'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    last_date = df['arrival_date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    future_numeric = (future_dates - df['arrival_date'].min()).days.values.reshape(-1, 1)
    future_poly = poly.transform(future_numeric)
    predictions = model.predict(future_poly)
    
    residuals = y - model.predict(X_poly)
    std_residuals = np.std(residuals)
    
    return future_dates, predictions, std_residuals

# ---------------------------
# USER INPUT
# ---------------------------
crop = st.selectbox("🌾 Select Crop", ["Wheat", "Rice", "Onion", "Potato"])
state = st.selectbox("📍 Select State", [
    "Gujarat", "Punjab", "Maharashtra", "Uttar Pradesh"
])

with st.spinner("Fetching mandis..."):
    mandi_list = get_mandis(state)

if not mandi_list:
    st.error("No mandis found for selected state")
    st.stop()

mandi = st.selectbox("🏪 Select Mandi", mandi_list)

with st.spinner("Loading data..."):
    df = load_data(crop, state, mandi)

if df.empty:
    st.error("No data available for selected combination")
    st.stop()

if len(df) < 30:
    st.error(f"Not enough data (minimum 30 records required, only {len(df)} found)")
    st.stop()

# ---------------------------
# LAST 30 DAYS TREND
# ---------------------------
st.subheader("📊 Last 30 Days Price Trend")
last_30 = df.tail(30)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(last_30['arrival_date'], last_30['modal_price'], marker='o', linewidth=2, color='blue')
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (₹)", fontsize=12)
ax.set_title(f"Price Trend for {crop} in {mandi}, {state}", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# FORECAST
# ---------------------------
with st.spinner("Generating forecast..."):
    future_dates, predictions, std_dev = forecast_prices(df, days=7)

min_price = (predictions - std_dev).min()
max_price = (predictions + std_dev).max()
confidence_width = max_price - min_price
confidence = "High" if confidence_width < 100 else "Medium" if confidence_width < 300 else "Low"

today_price = df['modal_price'].iloc[-1]
avg_future = predictions.mean()
price_change_percent = ((avg_future - today_price) / today_price) * 100

if avg_future > today_price:
    recommendation = "📈 Prices are likely to rise — consider holding stock for 3–5 days."
else:
    recommendation = "📉 Prices are likely to fall — consider selling now."

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
st.subheader("📅 7-Day Forecast")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔻 Expected Min Price", f"₹{round(min_price,2)}")
    st.metric("🔺 Expected Max Price", f"₹{round(max_price,2)}")
with col2:
    st.metric("📊 Confidence Level", confidence)
    st.metric("📈 Price Change", f"{price_change_percent:+.1f}%")
with col3:
    st.metric("💰 Current Price", f"₹{round(today_price,2)}")
    st.metric("📊 Avg Forecast Price", f"₹{round(avg_future,2)}")

st.subheader("📋 Detailed Forecast")
forecast_df = pd.DataFrame({
    'Date': future_dates.strftime('%Y-%m-%d'),
    'Predicted Price': predictions.round(2),
    'Lower Bound': (predictions - std_dev).round(2),
    'Upper Bound': (predictions + std_dev).round(2)
})
st.dataframe(forecast_df, use_container_width=True)

st.subheader("💡 Recommendation")
if avg_future > today_price:
    st.success(recommendation)
else:
    st.warning(recommendation)

# ---------------------------
# VISUALIZATION
# ---------------------------
st.subheader("📈 Price Forecast Visualization")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df['arrival_date'], df['modal_price'], label='Historical Prices', color='blue', alpha=0.7)
ax2.plot(future_dates, predictions, label='Forecast', color='red', linewidth=2)
ax2.fill_between(future_dates, predictions - std_dev, predictions + std_dev, color='red', alpha=0.2, label='Confidence Interval')
ax2.axvline(x=df['arrival_date'].iloc[-1], color='gray', linestyle='--', label='Today')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (₹)')
ax2.set_title(f'Price Forecast for {crop} in {mandi}, {state}')
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# ACCURACY
# ---------------------------
df['date_numeric'] = (df['arrival_date'] - df['arrival_date'].min()).dt.days
X_all = df['date_numeric'].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
X_all_poly = poly.fit_transform(X_all)
model = LinearRegression()
model.fit(X_all_poly, df['modal_price'].values)
pred_all = model.predict(X_all_poly)
mae = mean_absolute_error(df['modal_price'].values[-30:], pred_all[-30:])
mape = np.mean(np.abs((df['modal_price'].values[-30:] - pred_all[-30:]) / df['modal_price'].values[-30:])) * 100

st.subheader("📏 Model Accuracy Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Absolute Error (MAE)", f"₹{round(mae,2)}")
with col2:
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{round(mape,1)}%")

st.markdown("---")
st.caption("Note: Forecast uses polynomial regression (degree 2). "
           "Accuracy depends on data quality and market conditions. "
           "This tool is for informational purposes only.")
