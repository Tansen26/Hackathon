import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = "579b464db66ec23bdd0000017ce9d9b9e36c456f41f78d04d7c1e97d"
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

st.set_page_config(page_title="AI Mandi Predictor", layout="wide")
st.title("🌾 AI-Powered Mandi Price Forecast Tool")


# ---------------------------
# GET MANDIS
# ---------------------------
@st.cache_data(ttl=3600)
def get_mandis(state: str) -> list:
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[state]": state,
    }
    try:
        response = requests.get(URL, params=params, timeout=15)
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:
            return []
        df = pd.DataFrame(records)
        if "market" not in df.columns:
            return []
        return sorted(df["market"].dropna().unique().tolist())
    except Exception as e:
        st.error(f"Error fetching mandis: {e}")
        return []


# ---------------------------
# FETCH PRICE DATA
# ---------------------------
@st.cache_data(ttl=3600)
def load_data(crop: str, state: str, mandi: str) -> pd.DataFrame:
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[commodity]": crop,
        "filters[state]": state,
        "filters[market]": mandi,
    }
    try:
        response = requests.get(URL, params=params, timeout=15)
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df = df.dropna(subset=["modal_price", "arrival_date"])
        df = df.sort_values("arrival_date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


# ---------------------------
# FORECAST — returns model too so accuracy reuse is clean
# ---------------------------
def build_model_and_forecast(df: pd.DataFrame, days: int = 7):
    origin = df["arrival_date"].min()
    df = df.copy()
    df["date_numeric"] = (df["arrival_date"] - origin).dt.days

    X = df["date_numeric"].values.reshape(-1, 1)
    y = df["modal_price"].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # In-sample residuals for confidence interval
    residuals = y - model.predict(X_poly)
    std_dev = float(np.std(residuals))

    # Future dates
    last_date = df["arrival_date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    future_numeric = ((future_dates - origin).days).values.reshape(-1, 1)
    future_poly = poly.transform(future_numeric)
    predictions = model.predict(future_poly)

    # Accuracy on last 30 in-sample points
    tail_n = min(30, len(y))
    y_tail = y[-tail_n:]
    pred_tail = model.predict(X_poly)[-tail_n:]
    mae = mean_absolute_error(y_tail, pred_tail)
    mape = float(
        np.mean(np.abs((y_tail - pred_tail) / np.where(y_tail == 0, 1, y_tail))) * 100
    )

    return future_dates, predictions, std_dev, mae, mape


# ---------------------------
# USER INPUTS
# ---------------------------
crop = st.selectbox("🌾 Select Crop", ["Wheat", "Rice", "Onion", "Potato"])
state = st.selectbox(
    "📍 Select State", ["Gujarat", "Punjab", "Maharashtra", "Uttar Pradesh"]
)

with st.spinner("Fetching mandis..."):
    mandi_list = get_mandis(state)

if not mandi_list:
    st.error("No mandis found for the selected state. Try a different state.")
    st.stop()

mandi = st.selectbox("🏪 Select Mandi", mandi_list)

# ---------------------------
# LOAD DATA
# ---------------------------
with st.spinner("Loading price data..."):
    df = load_data(crop, state, mandi)

if df.empty:
    st.error("No data available for the selected combination. Try different filters.")
    st.stop()

if len(df) < 30:
    st.error(
        f"Not enough data to forecast (minimum 30 records required, only {len(df)} found). "
        "Try a different crop/state/mandi combination."
    )
    st.stop()

# ---------------------------
# LAST 30 DAYS TREND
# ---------------------------
st.subheader("📊 Last 30 Days Price Trend")
last_30 = df.tail(30)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    last_30["arrival_date"],
    last_30["modal_price"],
    marker="o",
    linewidth=2,
    color="steelblue",
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (₹)", fontsize=12)
ax.set_title(f"Price Trend — {crop} | {mandi}, {state}", fontsize=13)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# GENERATE FORECAST
# ---------------------------
with st.spinner("Generating 7-day forecast..."):
    future_dates, predictions, std_dev, mae, mape = build_model_and_forecast(df, days=7)

# ---------------------------
# METRICS
# ---------------------------
min_price = float((predictions - std_dev).min())
max_price = float((predictions + std_dev).max())
confidence_width = max_price - min_price
confidence = "High" if confidence_width < 100 else ("Medium" if confidence_width < 300 else "Low")

today_price = float(df["modal_price"].iloc[-1])
avg_future = float(predictions.mean())
price_change_pct = ((avg_future - today_price) / today_price) * 100

st.subheader("📅 7-Day Forecast Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔻 Expected Min Price", f"₹{min_price:,.2f}")
    st.metric("🔺 Expected Max Price", f"₹{max_price:,.2f}")
with col2:
    st.metric("📊 Confidence Level", confidence)
    st.metric("📈 Price Change", f"{price_change_pct:+.1f}%")
with col3:
    st.metric("💰 Current Price", f"₹{today_price:,.2f}")
    st.metric("📊 Avg Forecast Price", f"₹{avg_future:,.2f}")

# ---------------------------
# FORECAST TABLE
# ---------------------------
st.subheader("📋 Detailed Day-by-Day Forecast")
forecast_df = pd.DataFrame(
    {
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predicted Price (₹)": predictions.round(2),
        "Lower Bound (₹)": (predictions - std_dev).round(2),
        "Upper Bound (₹)": (predictions + std_dev).round(2),
    }
)
st.dataframe(forecast_df, use_container_width=True)

# ---------------------------
# RECOMMENDATION
# ---------------------------
st.subheader("💡 Recommendation")
if avg_future > today_price:
    st.success("📈 Prices are likely to rise — consider holding stock for 3–5 days.")
else:
    st.warning("📉 Prices are likely to fall — consider selling now.")

# ---------------------------
# FORECAST VISUALIZATION
# ---------------------------
st.subheader("📈 Price Forecast Visualization")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["arrival_date"], df["modal_price"], label="Historical Prices", color="steelblue", alpha=0.7)
ax2.plot(future_dates, predictions, label="Forecast", color="crimson", linewidth=2)
ax2.fill_between(
    future_dates,
    predictions - std_dev,
    predictions + std_dev,
    color="crimson",
    alpha=0.2,
    label="Confidence Interval",
)
ax2.axvline(x=df["arrival_date"].iloc[-1], color="gray", linestyle="--", label="Today")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (₹)")
ax2.set_title(f"Price Forecast — {crop} | {mandi}, {state}")
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ---------------------------
# MODEL ACCURACY
# ---------------------------
st.subheader("📏 Model Accuracy (Last 30 Data Points)")
c1, c2 = st.columns(2)
with c1:
    st.metric("Mean Absolute Error (MAE)", f"₹{mae:,.2f}")
with c2:
    st.metric("Mean Absolute % Error (MAPE)", f"{mape:.1f}%")

st.markdown("---")
st.caption(
    "Forecast uses polynomial regression (degree 2). "
    "Accuracy depends on data quality and market volatility. "
    "This tool is for informational purposes only."
)
