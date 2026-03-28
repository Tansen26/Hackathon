import requests
import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

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
        
        mandis = sorted(df['market'].dropna().unique())
        return mandis
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

        # CLEANING
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])

        df = df.dropna(subset=['modal_price'])
        df = df.sort_values('arrival_date')

        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

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

# ---------------------------
# LOAD DATA
# ---------------------------
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
ax.plot(last_30['arrival_date'], last_30['modal_price'], marker='o', linewidth=2)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (₹)", fontsize=12)
ax.set_title(f"Price Trend for {crop} in {mandi}, {state}", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# PREPARE FOR MODEL
# ---------------------------
model_df = df[['arrival_date', 'modal_price']].copy()
model_df.columns = ['ds', 'y']

# ---------------------------
# TRAIN MODEL
# ---------------------------
with st.spinner("Training forecast model..."):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(model_df)

# ---------------------------
# FORECAST 7 DAYS
# ---------------------------
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

future_7 = forecast.tail(7)

# ---------------------------
# PRICE RANGE + CONFIDENCE
# ---------------------------
min_price = future_7['yhat_lower'].min()
max_price = future_7['yhat_upper'].max()

confidence_width = max_price - min_price

if confidence_width < 100:
    confidence = "High"
    confidence_color = "green"
elif confidence_width < 300:
    confidence = "Medium"
    confidence_color = "orange"
else:
    confidence = "Low"
    confidence_color = "red"

# ---------------------------
# DECISION LOGIC
# ---------------------------
today_price = model_df['y'].iloc[-1]
avg_future = future_7['yhat'].mean()
price_change_percent = ((avg_future - today_price) / today_price) * 100

if avg_future > today_price:
    recommendation = "📈 Prices are likely to rise — consider holding stock for 3–5 days."
    recommendation_type = "hold"
else:
    recommendation = "📉 Prices are likely to fall — consider selling now."
    recommendation_type = "sell"

# ---------------------------
# OUTPUT
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

# Display forecast table
st.subheader("📋 Detailed Forecast")
forecast_display = future_7[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
forecast_display = forecast_display.round(2)
st.dataframe(forecast_display, use_container_width=True)

st.subheader("💡 Recommendation")
if recommendation_type == "hold":
    st.success(recommendation)
else:
    st.warning(recommendation)

# ---------------------------
# FORECAST VISUALIZATION
# ---------------------------
st.subheader("📈 Price Forecast Visualization")

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(model_df['ds'], model_df['y'], label='Historical Prices', color='blue', alpha=0.7)
ax2.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red', linewidth=2)
ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                  color='red', alpha=0.2, label='Confidence Interval')
ax2.axvline(x=model_df['ds'].iloc[-1], color='gray', linestyle='--', label='Today')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (₹)')
ax2.set_title(f'Price Forecast for {crop} in {mandi}, {state}')
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# MODEL ACCURACY
# ---------------------------
actual = model_df['y'].tail(30)
pred = forecast['yhat'].tail(30)

mae = np.mean(np.abs(actual.values - pred.values))
mape = np.mean(np.abs((actual.values - pred.values) / actual.values)) * 100

st.subheader("📏 Model Accuracy Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Absolute Error (MAE)", f"₹{round(mae,2)}")
with col2:
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{round(mape,1)}%")

# Additional information
st.markdown("---")
st.caption("Note: Forecast accuracy depends on data quality and market conditions. "
           "This tool is for informational purposes only.")
