# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import zipfile
import os

# ------------------------
# Ensure models directory exists
# ------------------------
os.makedirs("models_zip", exist_ok=True)

# ------------------------
# Extract zip files (repo paths)
# ------------------------
with zipfile.ZipFile("rf_model.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")

with zipfile.ZipFile("scaler.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")

# ------------------------
# Load models and scaler
# ------------------------
with open("models_zip/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models_zip/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Already unzipped files
ann_model = load_model("ann_model.keras")

with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will churn using Logistic Regression, Random Forest, or ANN.")
st.markdown("---")

# ------------------------
# Sidebar: User Inputs
# ------------------------
st.sidebar.header("Input Customer Features")
input_data = {}

input_data['age'] = st.sidebar.number_input("Age (years)", 18, 100, 30)
input_data['account_tenure'] = st.sidebar.number_input("Account Tenure (months)", 0, 600, 36)
input_data['credit_cards'] = st.sidebar.number_input("Credit Cards Count", 0, 20, 1)
input_data['installment_loans'] = st.sidebar.number_input("Installment Loans Count", 0, 10, 0)
input_data['ccfp_products'] = st.sidebar.number_input("CCFP Products Count", 0, 10, 1)
input_data['package_type'] = st.sidebar.selectbox("Package Type", ["Basic", "Silver", "Gold", "Platinum"])
input_data['atm_trans_count'] = st.sidebar.number_input("ATM Transactions (3 months)", 0, 1000, 10)
input_data['total_trans_count'] = st.sidebar.number_input("Total Transactions (3 months)", 0, 5000, 50)
input_data['mobile_banking_pct'] = st.sidebar.number_input("Mobile Banking Usage (%)", 0, 100, 20)

# Encode package_type
package_dict = {'Basic':0, 'Silver':0, 'Gold':0, 'Platinum':1}
input_data_encoded = input_data.copy()
input_data_encoded['package_type'] = package_dict[input_data['package_type']]

# Convert to DataFrame
input_df = pd.DataFrame([input_data_encoded])

# Scale features
scaled_input = scaler.transform(input_df)

# Display input
st.subheader("📊 User Input Features")
st.write(input_df)
st.markdown("---")

# ------------------------
# Model selection
# ------------------------
model_choice = st.selectbox(
    "Select Model for Prediction:",
    ["Logistic Regression", "Random Forest", "ANN"]
)

# ------------------------
# Predict Button
# ------------------------
if st.button("🔍 Predict"):
    if model_choice == "Logistic Regression":
        pred = log_model.predict(scaled_input)[0]
        prob = log_model.predict_proba(scaled_input)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(scaled_input)[0]
        prob = rf_model.predict_proba(scaled_input)[0]
    else:  # ANN
        pred = (ann_model.predict(scaled_input) > 0.5).astype(int)[0][0]
        prob_ann = ann_model.predict(scaled_input)[0][0]
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
