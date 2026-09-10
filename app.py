import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

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
st.markdown("Real-World 3D Architectural City Mapping & AI Risk Prediction.")

col1, col2 = st.columns(2)
with col1:
    time_of_day = st.slider("Simulate Time of Day (Hours)", 0.0, 24.0, 18.0)
with col2:
    vibration = st.slider("Simulate Bridge Vibration (Hz)", 0.1, 5.0, 3.5)

# Generate sensor points across Bengaluru
num_sensors = 40
latitudes = np.random.uniform(12.93, 13.01, num_sensors)
longitudes = np.random.uniform(77.55, 77.65, num_sensors)

current_conditions = pd.DataFrame({
    'time_of_day': [time_of_day]*num_sensors, 
    'bridge_vibration': [vibration]*num_sensors
})
risk_scores = model.predict(current_conditions)

# Data points for AI risk spikes overlaying the city
risk_data = pd.DataFrame({
    'lat': latitudes,
    'lon': longitudes,
    'risk': risk_scores,
    'elevation': risk_scores * 20,
    'color': [[255, 50, 50, 200] if r > 75 else [50, 200, 50, 150] for r in risk_scores]
})

# Render 3D Map using OpenStreetMap / Carto basemap for realistic urban layout
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v10',
    initial_view_state=pdk.ViewState(
        latitude=12.9716, 
        longitude=77.5946, 
        zoom=13, 
        pitch=55,
        bearing=30
    ),
    layers=[
        # AI Risk Spikes Layer
        pdk.Layer(
            'ColumnLayer',
            data=risk_data,
            get_position='[lon, lat]',
            get_elevation='elevation',
            elevation_scale=5,
            radius=150,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        ),
    ],
))

if st.button("Generate Incident Report"):
    critical_nodes = len(risk_data[risk_data['risk'] > 75])
    st.error(f"⚠️ {critical_nodes} critical infrastructure nodes flagged across the urban grid.")
