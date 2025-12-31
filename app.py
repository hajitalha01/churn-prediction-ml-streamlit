# -----------------------------
# Final app.py
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import zipfile
import os

# -----------------------------
# Extract zip files (if any)
# -----------------------------
os.makedirs("models_zip", exist_ok=True)

with zipfile.ZipFile("rf_model.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")

with zipfile.ZipFile("scaler.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")

# -----------------------------
# Load models and scaler
# -----------------------------
with open("models_zip/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models_zip/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

ann_model = load_model("ann_model.keras")

with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will churn using Logistic Regression, Random Forest, or ANN.")
st.markdown("---")

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("Input Customer Features")
input_data = {}

for col in feature_cols:
    if col == "package_type":
        input_data[col] = st.sidebar.selectbox(
            "Package Type",
            ["Basic", "Silver", "Gold", "Platinum"]
        )
    else:
        input_data[col] = st.sidebar.number_input(
            col.replace("_", " ").title(),
            min_value=0.0, max_value=100000.0, value=0.0
        )

# Encode package_type
package_dict = {"Basic":0, "Silver":0, "Gold":0, "Platinum":1}
input_data_encoded = input_data.copy()
input_data_encoded["package_type"] = package_dict[input_data["package_type"]]

# Convert to DataFrame in correct order
input_df = pd.DataFrame([input_data_encoded])
input_df = input_df[feature_cols]

# Scale features
scaled_input = scaler.transform(input_df)

# Display input
st.subheader("📊 User Input Features")
st.write(input_df)
st.markdown("---")

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.selectbox(
    "Select Model for Prediction:",
    ["Logistic Regression", "Random Forest", "ANN"]
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict"):
    if model_choice == "Logistic Regression":
        pred = log_model.predict(scaled_input)[0]
        prob = log_model.predict_proba(scaled_input)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(scaled_input)[0]
        prob = rf_model.predict_proba(scaled_input)[0]
    else:  # ANN
        prob_ann = ann_model.predict(scaled_input)[0][0]
        pred = 1 if prob_ann > 0.5 else 0
        prob = [1-prob_ann, prob_ann]

    # Display prediction
    st.subheader("🧠 Prediction Result")
    result_text = "Churn ⚠️" if pred==1 else "No Churn ✅"
    st.success(result_text) if pred==0 else st.error(result_text)

    # Display prediction probability
    st.subheader("📈 Prediction Probability")
    st.write(f"No Churn : {prob[0]:.2%}")
    st.write(f"Churn    : {prob[1]:.2%}")

st.caption("Developed using Streamlit & ML Models (Logistic Regression, Random Forest, ANN)")
