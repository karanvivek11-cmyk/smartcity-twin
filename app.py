import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

# Train a lightweight AI model on the fly
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
st.markdown("Live infrastructure AI predictions and 3D geospatial mapping.")

col1, col2 = st.columns(2)
with col1:
    time_of_day = st.slider("Simulate Time of Day (Hours)", 0.0, 24.0, 18.0)
with col2:
    vibration = st.slider("Simulate Bridge Vibration (Hz)", 0.1, 5.0, 3.5)

num_sensors = 50
# Generating synthetic data around Bengaluru
latitudes = np.random.uniform(12.85, 13.05, num_sensors)
longitudes = np.random.uniform(77.45, 77.75, num_sensors)

current_conditions = pd.DataFrame({
    'time_of_day': [time_of_day]*num_sensors, 
    'bridge_vibration': [vibration]*num_sensors
})
risk_scores = model.predict(current_conditions)
risk_scores = np.clip(risk_scores + np.random.normal(0, 10, num_sensors), 0, 100)

map_data = pd.DataFrame({
    'lat': latitudes,
    'lon': longitudes,
    'risk': risk_scores,
    'color_r': np.where(risk_scores > 75, 255, 0),
    'color_g': np.where(risk_scores <= 75, 255, 0),
    'color_b': 0,
    'elevation': risk_scores * 30
})

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10',
    initial_view_state=pdk.ViewState(
        latitude=12.9716, 
        longitude=77.5946, 
        zoom=10, 
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'ColumnLayer',
            data=map_data,
            get_position='[lon, lat]',
            get_elevation='elevation',
            elevation_scale=10,
            radius=400,
            get_fill_color='[color_r, color_g, color_b, 160]',
            pickable=True,
            auto_highlight=True,
        ),
    ],
))

if st.button("Generate Incident Report"):
    critical_nodes = len(map_data[map_data['risk'] > 75])
    st.error(f"⚠️ {critical_nodes} critical infrastructure nodes detected requiring immediate inspection.")
