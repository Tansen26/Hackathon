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
    
    response = requests.get(URL, params=params)
    data = response.json()
    
    df = pd.DataFrame(data.get('records', []))
    
    if df.empty:
        return []
    
    mandis = sorted(df['market'].dropna().unique())
    return mandis

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
    
    response = requests.get(URL, params=params)
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

# ---------------------------
# USER INPUT
# ---------------------------
crop = st.selectbox("🌾 Select Crop", ["Wheat", "Rice", "Onion", "Potato"])

state = st.selectbox("📍 Select State", [
    "Gujarat", "Punjab", "Maharashtra", "Uttar Pradesh"
])

mandi_list = get_mandis(state)

if not mandi_list:
    st.error("No mandis found for selected state")
    st.stop()

mandi = st.selectbox("🏪 Select Mandi", mandi_list)

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_data(crop, state, mandi)

if df.empty:
    st.error("No data available for selected combination")
    st.stop()

if len(df) < 30:
    st.error("Not enough data (minimum 30 records required)")
    st.stop()

# ---------------------------
# LAST 30 DAYS TREND
# ---------------------------
st.subheader("📊 Last 30 Days Price Trend")

last_30 = df.tail(30)

plt.figure()
plt.plot(last_30['arrival_date'], last_30['modal_price'])
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price (₹)")
st.pyplot(plt)

# ---------------------------
# PREPARE FOR MODEL
# ---------------------------
model_df = df[['arrival_date', 'modal_price']]
model_df.columns = ['ds', 'y']

# ---------------------------
# TRAIN MODEL
# ---------------------------
model = Prophet()
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
elif confidence_width < 300:
    confidence = "Medium"
else:
    confidence = "Low"

# ---------------------------
# DECISION LOGIC
# ---------------------------
today_price = model_df['y'].iloc[-1]
avg_future = future_7['yhat'].mean()

if avg_future > today_price:
    recommendation = "📈 Prices are likely to rise — consider holding stock for 3–5 days."
else:
    recommendation = "📉 Prices are likely to fall — consider selling now."

# ---------------------------
# OUTPUT
# ---------------------------
st.subheader("📅 7-Day Forecast")

col1, col2 = st.columns(2)

with col1:
    st.metric("🔻 Min Price", f"₹{round(min_price,2)}")
    st.metric("🔺 Max Price", f"₹{round(max_price,2)}")

with col2:
    st.metric("📊 Confidence", confidence)

st.subheader("💡 Recommendation")
st.success(recommendation)

# ---------------------------
# MODEL ACCURACY
# ---------------------------
actual = model_df['y'].tail(30)
pred = forecast['yhat'].tail(30)

mae = np.mean(np.abs(actual.values - pred.values))

st.subheader("📏 Model Accuracy")
st.write(f"MAE: {round(mae,2)}")