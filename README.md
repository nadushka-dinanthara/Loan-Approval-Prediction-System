# Loan Approval Prediction System

A machine learning-based system to predict whether a loan will be approved or rejected. This project uses **Random Forest and Logistic Regression models**, and provides a **user-friendly dashboard using Streamlit** for entering loan applicant details and getting immediate predictions.

---

## Features

- **Predict loan approval** based on applicant’s financial and personal details  
- **Random Forest model** with class weighting for imbalanced dataset  
- **Custom probability threshold** to reduce false positives  
- **Interactive Streamlit dashboard** with sliders, dropdowns, and number inputs  
- **Probability visualization** using a progress bar  
- Preprocessing handles categorical encoding and column cleaning automatically  

---

## Dataset

The dataset should include the following columns:

- `loan_id`  
- `no_of_dependents`  
- `education` (Graduate / Not Graduate / Unknown)  
- `self_employed` (Yes / No)  
- `income_annum` (Annual income in Rs.)  
- `loan_amount`  
- `loan_term` (in months)  
- `cibil_score` (300–900)  
- `residential_assets_value`  
- `commercial_assets_value`  
- `luxury_assets_value`  
- `bank_asset_value`  
- `loan_status` (Approved / Rejected)  

> The system automatically drops `loan_id` during training and preprocessing.

---

## Project Structure

