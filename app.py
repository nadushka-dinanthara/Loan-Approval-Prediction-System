import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_features


# Load the trained Random Forest model

rf_model = joblib.load("models/random_forest_model.pkl")
threshold = 0.55  # custom threshold for predicting Approved/Rejected

# Streamlit page config

st.set_page_config(page_title="Loan Approval Dashboard", page_icon="💰", layout="centered")
st.title("💰 Loan Approval Prediction Dashboard")
st.write("Enter the loan applicant's details below to predict whether the loan will be approved or rejected.")


# Input form

with st.form(key="loan_form"):
    col1, col2 = st.columns(2)

    # Left column inputs
    no_of_dependents = col1.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = col1.selectbox("Education", ["Graduate", "Not Graduate", "Unknown"])
    self_employed = col1.selectbox("Self Employed", ["Yes", "No"])
    income_annum = col1.number_input("Annual Income (Rs.)", min_value=0, value=500000)
    commercial_assets_value = col1.number_input("Commercial Assets Value (Rs.)", min_value=0, value=0)
    luxury_assets_value = col1.number_input("Luxury Assets Value (Rs.)", min_value=0, value=0)

    # Right column inputs
    loan_amount = col2.number_input("Loan Amount (Rs.)", min_value=0, value=100000)
    loan_term = col2.number_input("Loan Term (months)", min_value=6, value=12)
    cibil_score = col2.slider("CIBIL Score", min_value=300, max_value=900, value=650)
    residential_assets_value = col2.number_input("Residential Assets Value (Rs.)", min_value=0, value=200000)
    bank_asset_value = col2.number_input("Bank Asset Value (Rs.)", min_value=0, value=100000)

    submit_button = st.form_submit_button(label="Predict Loan Approval")


# Make prediction

if submit_button:
    # Convert inputs to DataFrame
    new_loan = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    # Preprocess features (cleaning + encoding + drop loan_id if exists)
    new_loan = preprocess_features(new_loan)

    # Predict probability for Approved (class 1)
    prob = rf_model.predict_proba(new_loan)[:, 1][0]
    prediction = "Approved" if prob >= threshold else "Rejected"

    # Display results
    st.markdown("### Prediction Result")
    if prediction == "Approved":
        st.success(f"✅ Loan Approved! Probability: {prob:.2f}")
    else:
        st.error(f"❌ Loan Rejected! Probability: {prob:.2f}")

    # Progress bar for probability
    st.progress(prob)
