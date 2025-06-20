# minimal_app.py
import streamlit as st
import numpy as np

st.title("Gwamz Stream Predictor")
st.write("Lite version while full app builds")

release_date = st.date_input("Release Date")
explicit = st.checkbox("Explicit Content")
version_type = st.selectbox("Version", ["Original", "Sped Up", "Remix"])
tracks = st.slider("Tracks in Album", 1, 20, 3)

if st.button("Predict"):
    # Simple prediction formula
    base = 500000
    month_factor = 0.8 + (release_date.month / 12)
    version_factor = 1.3 if "Sped" in version_type else 1.0
    explicit_factor = 1.2 if explicit else 1.0
    
    prediction = int(base * month_factor * version_factor * explicit_factor)
    st.success(f"Estimated Streams: {prediction:,}")