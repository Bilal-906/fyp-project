# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and scaler
model = load_model("model.h5")  # Save your model in this file
scaler = joblib.load("scaler.save")  # Save your fitted scaler

# Title
st.title("🔐 AMBER Threat Detection Dashboard")

# Input fields
st.header("📥 Input Features")
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# Add as many as your model requires

# Submit button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])  # add all features in order
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Threat Detected" if prediction[0][0] > 0.5 else "No Threat"
    st.success(f"🚨 Result: {result}")
