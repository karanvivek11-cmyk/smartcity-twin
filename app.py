import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble]. RandomForestRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor

# Put your permanent Mapbox public token here (starts with pk.eyJ...)
MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN_HERE"

@st.cache_resource
def train_model():
    np.random.seed(42)
    X = pd.DataFrame({
        'time_of_day': np.random.uniform(0, 24, 1000),
        'bridge_vibration': np.random.uniform(0.1, 5.0, 1000)
    })
    y = (X['time_of_day'] * 2 + X['bridge_vibration'] * 10).clip(0, 100)
    model = RandomForestRegressor(n_estimators=10, max_depth=5)
    model.fit(X, y)
    return model

model = train_model()

st.title("🏙️ Smart City Digital Twin")
st.markdown("Bengaluru Urban & Rural Satellite Telemetry")

col1, col2 = st.columns(2)
with col1:
    time_of_day = st.slider("Simulate Time of Day (Hours)", 0.0, 24.0, 18.0)
with col2:
    vibration = st.slider("Simulate Bridge Vibration (Hz)", 0.1, 5.0, 3.5)

num_sensors = 150
latitudes = np.random.uniform(12.70, 13.35, num_sensors)
longitudes = np.random.uniform(77.25, 77.90, num_sensors)

current_conditions = pd.DataFrame({
    'time_of_day': [time_of_day]*num_sensors, 
    'bridge_vibration': [vibration]*num_sensors
})
risk_scores = model.predict(current_conditions)

risk_data = pd.DataFrame({
    'lat': latitudes,
    'lon': longitudes,
    'risk': risk_scores,
    'elevation': risk_scores * 30,
    'color': [[255, 50, 50, 230] if r > 75 else [0, 255, 120, 200] for r in risk_scores]
})

st.pydeck_chart(pdk.Deck(
    map_provider='mapbox',
    map_style='mapbox://styles/mapbox/satellite-v9',
    api_keys={'mapbox': MAPBOX_TOKEN},
    initial_view_state=pdk.ViewState(
        latitude=13.02, 
        longitude=77.59, 
        zoom=9.5, 
        pitch=40,
        bearing=10
    ),
    layers=[
        pdk.Layer(
            'ColumnLayer',
            data=risk_data,
            get_position='[lon, lat]',
            get_elevation='elevation',
            elevation_scale=5,
            radius=400,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        ),
    ],
))

if st.button("Generate Incident Report"):
    critical_nodes = len(risk_data[risk_data['risk'] > 75])
    st.error(f"⚠️ {critical_nodes} critical infrastructure nodes flagged on satellite telemetry.")
