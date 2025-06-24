import streamlit as st
st.set_page_config(page_title="MAGIC Gamma Classification", layout="centered")

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_nn_model.h5')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler1.pkl')

model = load_model()
scaler = load_scaler()

st.title("🔮 MAGIC Gamma Ray Event Classifier")

st.markdown("""
This app predicts whether a gamma-ray event is a **signal (g)** or **background noise (h)** 
using a neural network trained on the MAGIC Gamma Telescope dataset.
You can enter values manually or upload a CSV file with 10 feature columns.
""")

# Feature input
feature_names = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'
]

input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

# Manual Entry
if input_method == "Manual Entry":
    values = [st.slider(f"{name}", 0.0, 1000.0, step=0.1) for name in feature_names]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = model.predict(X_scaled)[0][0]
    label = "g (Gamma Ray)" if proba > 0.5 else "h (Hadron)"

    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{proba:.2f}`")
    st.progress(int(proba * 100))

# CSV Upload
else:
    uploaded = st.file_uploader("Upload a CSV with 10 feature columns:", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        if not set(feature_names).issubset(set(df.columns)):
            st.error("CSV must contain these columns:\n" + ", ".join(feature_names))
        else:
            df = df[feature_names]  # Reorder columns safely
            X_scaled = scaler.transform(df.values)
            predictions = model.predict(X_scaled).ravel()

            df['Prediction'] = ['g (Gamma Ray)' if p > 0.5 else 'h (Hadron)' for p in predictions]
            df['Probability'] = predictions
            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")


