import os
import io
import time
import json
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st 
from datetime import datetime
from streamlit_folium import folium_static
import folium

# ------------------ CONFIG ------------------
st.set_page_config(page_title=" Kisan Crop Recommender", layout="wide",)

FEATURE_COLS = ["N","P","K","temperature","humidity","ph","rainfall"]

# Secrets (optional)
OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", None)

# ------------------ STYLES ------------------
st.markdown(f"""
<style>
.stApp {{
   background-color:#000000;
   color: white;
   front-family:'Segoe UI',sans-serif;
 
   
}}
/* translucent card for readability */
.block-container {{
    background-color: rgba(0,0,0,0.90);
    padding: 1.25rem 1.25rem 2rem 1.25rem;
    border-radius: 14px;
    color:white;
}}
h1, h2, h3,h4,h5,h6,p,label,span {{ color:white !importabt }}
.stButton > button {{
    background-color:#444444; color:white; font-weight:600;
    border-radius:10px; padding:8px 20px;
    border: 1px solid #888;
}}
.stButton > button:hover {{ background-color:#666666; }}
.result-box {{
    background: #222222; color:#00ff00; border:2px solid #00ff00;
    padding:14px; border-radius:12px; text-align:center; font-weight:800; font-size:22px;
}}
.small-muted {{ color:#aaa; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

# ------------------ HELPERS ------------------
@st.cache_resource
def load_model():
    with open("crop_model.pkl", "rb") as f:
        return pickle.load(f)

def predict_single(model, vals):
    """vals = [N,P,K,temp,humidity,ph,rainfall]"""
    X = np.array([vals], dtype=float)
    return model.predict(X)[0]

def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    return missing

def append_log(row: dict, path="predictions_log.csv"):
    df = pd.DataFrame([row])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False, mode="a", header=False)

def get_weather_by_coords(lat: float, lon: float, api_key: str | None):
    """Fetch temp/humidity/rainfall (mm) using OpenWeather current weather API."""
    # If no key, gracefully handle
    if not api_key:
        return {"error":"No OpenWeather API key found in secrets."}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units":"metric"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return {"error": f"API error: {r.status_code}"}
    data = r.json()
    temp = data.get("main", {}).get("temp")
    humidity = data.get("main", {}).get("humidity")
    # rainfall could be in last 1h or 3h
    rain = data.get("rain", {})
    rainfall = rain.get("1h") or rain.get("3h") or 0.0
    return {"temperature": temp, "humidity": humidity, "rainfall": rainfall}

# ------------------ LOAD MODEL ------------------
try:
    model = load_model()
except Exception as e:
    st.error(" Model load failed. Make sure `crop_model.pkl` is in the same folder.")
    st.stop()

# ------------------ HEADER ------------------
st.markdown("Crop Recommendation ")
st.markdown("<p class='small-muted'>Single prediction, bulk prediction, rainfall map & logs — all in one app.</p>", unsafe_allow_html=True)
st.write("---")

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs([" Single Prediction", " Bulk Prediction (CSV)", " Rainfall & Map", " Logs / Download"])

# ---------- TAB 1: SINGLE ----------
with tab1:
    st.subheader("Enter Soil & Weather Inputs")
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1)
        P = st.number_input("Phosphorus (P)", min_value=0.0, step=0.1)
        K = st.number_input("Potassium (K)", min_value=0.0, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (°C)", step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
        ph = st.number_input("Soil pH", min_value=0.0, step=0.1)
    with col3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button(" Recommend Crop", use_container_width=True)

    if go:
        pred = predict_single(model, [N, P, K, temperature, humidity, ph, rainfall])
        st.markdown(f"<div class='result-box'> Recommended Crop: {pred}</div>", unsafe_allow_html=True)

        # store prediction in a log
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "N": N, "P": P, "K": K,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
            "prediction": pred
        }
        append_log(row)
        st.caption("Saved to predictions_log.csv")

# ---------- TAB 2: BULK ----------
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    st.caption("CSV must contain columns: " + ", ".join(FEATURE_COLS))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        missing = validate_columns(df)
        if missing:
            st.error(f"These required columns are missing: {missing}")
        else:
            with st.spinner("Predicting..."):
                # predict for all rows
                X = df[FEATURE_COLS].values
                preds = model.predict(X)
                df_out = df.copy()
                df_out["prediction"] = preds
                time.sleep(0.3)

            st.success(f"Done! Predicted {len(df_out)} rows.")
            st.dataframe(df_out, use_container_width=True)

            # download button
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(" Download Results CSV", data=csv_bytes,
                               file_name="bulk_predictions.csv", mime="text/csv")

# ---------- TAB 3: MAP & RAINFALL ----------

with tab3:
    st.subheader("Rainfall & Map (Optional automation)")
    st.caption("Enter coordinates. If you have OPENWEATHER_KEY, we’ll fetch live rainfall.")
    with st.expander("Enter coordinates (recommended)"):
        lat = st.number_input("Latitude", value=25.0, format="%.6f")
        lon = st.number_input("Longitude", value=82.0, format="%.6f")
        fetch_weather = st.button("Get Weather & Show Map")

    if fetch_weather:
        # 1️ Fetch weather
        info = get_weather_by_coords(lat, lon, OPENWEATHER_KEY)
        if isinstance(info, dict) and "error" in info:
            st.warning(info["error"])
        else:
            st.success(
                f"Temp: {info['temperature']}°C | "
                f"Humidity: {info['humidity']}% | "
                f"Rainfall (last hr/3hr): {info['rainfall']} mm"
            )

        # 2️ Build folium map
        fmap = folium.Map(location=[lat, lon], zoom_start=8, tiles="OpenStreetMap")
        folium.Marker([lat, lon], tooltip="Your location").add_to(fmap)

        # 3️ Add precipitation layer without lambdas
        if OPENWEATHER_KEY:
            tile_url = (
                f"https://tile.openweathermap.org/map/precipitation_new/"
                "{z}/{x}/{y}.png"
                f"?appid={OPENWEATHER_KEY}"
            )
            folium.TileLayer(
                tiles=tile_url,
                name="Precipitation (OpenWeather)",
                attr="OpenWeatherMap",
                overlay=True,
                control=True,
                opacity=0.6
            ).add_to(fmap)

        # 4️ Add layer control then render
        folium.LayerControl().add_to(fmap)
        folium_static(fmap, width=900, height=500)



# ---------- TAB 4: LOGS ----------
with tab4:
    st.subheader("Prediction Logs")
    path = "predictions_log.csv"
    if os.path.exists(path):
        df_log = pd.read_csv(path)
        st.dataframe(df_log.tail(200), use_container_width=True)
        dl = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(" Download Full Log", data=dl, file_name="predictions_log.csv", mime="text/csv")
    else:
        st.info("No logs yet. Run a prediction from Tab 1.")


