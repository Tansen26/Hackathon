import os
import requests
import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
# For Streamlit Cloud: use st.secrets
# For other environments: use environment variable
if "API_KEY" in st.secrets:
    API_KEY = st.secrets["API_KEY"]
else:
    API_KEY = os.environ.get("API_KEY", "your-default-key")

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
# FETCH DATA FUNCTION (IMPROVED)
# ---------------------------
@st.cache_data
def load_data(crop, state, mandi):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[state]": state,
        "filters[commodity]": crop,
        "filters[market]": mandi
    }

    try:
        response = requests.get(URL, params=params, timeout=10)
        response.raise_for_status()   # Raise exception for HTTP errors
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return pd.DataFrame()

    data = response.json()

    # Check if API returned an error
    if data.get("status") != "ok":
        st.error(f"API error: {data.get('message', 'Unknown error')}")
        st.json(data)   # Show the full response for debugging
        return pd.DataFrame()

    if not data.get("records"):
        st.warning("No data found for this combination. Try another state/crop/mandi.")
        return pd.DataFrame()

    df = pd.DataFrame(data["records"])

    # Ensure required columns exist
    required = ["modal_price", "arrival_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in response: {missing}")
        st.write("Available columns:", list(df.columns))
        return pd.DataFrame()

    # Clean data
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    # The API returns date in dd/mm/yyyy format
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["modal_price", "arrival_date"])
    df = df.sort_values("arrival_date")

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
