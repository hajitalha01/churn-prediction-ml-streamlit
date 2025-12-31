# app.py
import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import zipfile

# -------------------------------
# Extract zip files (RF model + Scaler)
# -------------------------------
with zipfile.ZipFile("rf_model.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")
with zipfile.ZipFile("scaler.zip", 'r') as zip_ref:
    zip_ref.extractall("models_zip")

# -------------------------------
# Load models and scaler
# -------------------------------
with open("models_zip/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models_zip/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)
ann_model = load_model("ann_model.keras")

# -------------------------------
# Feature columns
# -------------------------------
all_features = [
    'customer_id','age','account_tenure_months','credit_cards_count',
    'installment_loans_count','ccfp_products_count','atm_trans_count_3m',
    'atm_trans_amount_3m','total_trans_count_3m','total_trans_amount_3m',
    'mobile_banking_usage_pct','avg_current_balance','avg_payment_balance',
    'credit_card_balance_change_3m','credit_card_turnover','payment_turnover',
    'marital_status_C','marital_status_D','marital_status_M','marital_status_N',
    'marital_status_T','marital_status_Unknown','marital_status_V','marital_status_W',
    'marital_status_d','marital_status_m','marital_status_t','marital_status_v',
    'marital_status_w','education_AC','education_AV','education_E','education_H',
    'education_HH','education_HI','education_I','education_S','education_SS',
    'education_UH','education_US','education_Unknown','education_a','education_e',
    'education_h','education_i','education_s','package_type_102','package_type_103',
    'package_type_104','package_type_105','package_type_107','package_type_108',
    'package_type_109','package_type_301','package_type_K01','package_type_M01',
    'package_type_O01'
]

# -------------------------------
# Streamlit Page
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will churn using Logistic Regression, Random Forest, or ANN.")
st.markdown("---")

# -------------------------------
# Sidebar: User Input
# -------------------------------
st.sidebar.header("Input Customer Features")

user_input = {}
user_input['customer_id'] = st.sidebar.number_input("Customer ID", 0, 1000000, 0)
user_input['age'] = st.sidebar.number_input("Age (years)", 18, 100, 30)
user_input['account_tenure_months'] = st.sidebar.number_input("Account Tenure (months)", 0, 600, 36)
user_input['credit_cards_count'] = st.sidebar.number_input("Credit Cards Count", 0, 20, 2)
user_input['installment_loans_count'] = st.sidebar.number_input("Installment Loans Count", 0, 10, 1)
user_input['ccfp_products_count'] = st.sidebar.number_input("CCFP Products Count", 0, 10, 1)
user_input['atm_trans_count_3m'] = st.sidebar.number_input("ATM Transactions (3 months)", 0, 1000, 10)
user_input['atm_trans_amount_3m'] = st.sidebar.number_input("ATM Transaction Amount (3 months)", 0, 100000, 500)
user_input['total_trans_count_3m'] = st.sidebar.number_input("Total Transactions (3 months)", 0, 5000, 50)
user_input['total_trans_amount_3m'] = st.sidebar.number_input("Total Transaction Amount (3 months)", 0, 100000, 2000)
user_input['mobile_banking_usage_pct'] = st.sidebar.number_input("Mobile Banking Usage (%)", 0, 100, 20)
user_input['avg_current_balance'] = st.sidebar.number_input("Average Current Balance", 0, 1000000, 5000)
user_input['avg_payment_balance'] = st.sidebar.number_input("Average Payment Balance", 0, 1000000, 2000)
user_input['credit_card_balance_change_3m'] = st.sidebar.number_input("Credit Card Balance Change (3 months)", 0, 100000, 500)
user_input['credit_card_turnover'] = st.sidebar.number_input("Credit Card Turnover", 0, 100000, 1000)
user_input['payment_turnover'] = st.sidebar.number_input("Payment Turnover", 0, 100000, 800)

# Categorical
user_input['marital_status'] = st.sidebar.selectbox("Marital Status", ["C","D","M","N","T","Unknown","V","W","d","m","t","v","w"])
user_input['education'] = st.sidebar.selectbox("Education", ["AC","AV","E","H","HH","HI","I","S","SS","UH","US","Unknown","a","e","h","i","s"])
user_input['package_type'] = st.sidebar.selectbox("Package Type", ["102","103","104","105","107","108","109","301","K01","M01","O01"])

# -------------------------------
# Prepare Input Data
# -------------------------------
input_encoded = {col:0 for col in all_features}

# Fill numeric values
for key in user_input:
    if key in input_encoded:
        input_encoded[key] = user_input[key]

# One-hot encode categorical
input_encoded[f"marital_status_{user_input['marital_status']}"] = 1
input_encoded[f"education_{user_input['education']}"] = 1
input_encoded[f"package_type_{user_input['package_type']}"] = 1

# DataFrame
input_df = pd.DataFrame([input_encoded], columns=all_features)

# Scale features
scaled_input = scaler.transform(input_df)

# -------------------------------
# Model selection
# -------------------------------
model_choice = st.selectbox("Select Model for Prediction", ["Logistic Regression","Random Forest","ANN"])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict"):
    if model_choice == "Logistic Regression":
        pred = log_model.predict(scaled_input)[0]
        prob = log_model.predict_proba(scaled_input)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(scaled_input)[0]
        prob = rf_model.predict_proba(scaled_input)[0]
    else:  # ANN
        pred_prob = ann_model.predict(scaled_input)[0][0]
        pred = int(pred_prob > 0.5)
        prob = [1 - pred_prob, pred_prob]

    # Display prediction
    st.subheader("🧠 Prediction Result")
    if pred == 0:
        st.success("No Churn ✅")
    else:
        st.error("Churn ⚠️")

    # Display probabilities
    st.subheader("📈 Prediction Probability")
    st.write(f"No Churn : {prob[0]:.2%}")
    st.write(f"Churn    : {prob[1]:.2%}")

st.caption("Developed using Streamlit & ML Models (Logistic Regression, Random Forest, ANN)")
