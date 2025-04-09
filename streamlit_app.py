import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, features
model = joblib.load("app/model.pkl")
scaler = joblib.load("app/scaler.pkl")
features = joblib.load("app/features.pkl")

st.title("💼 Employee Salary Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor's degree", "Master's degree", "PhD"])
job_title = st.selectbox("Job Title", ["Manager", "Engineer", "Analyst", "Administrator"])
experience = st.slider("Years of Experience", 0, 40, 5)

# Predict button
if st.button("Predict Salary"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Scale numerical features
    input_encoded[["Age", "Years of Experience"]] = scaler.transform(
        input_encoded[["Age", "Years of Experience"]])

    # Predict
    salary = model.predict(input_encoded)[0]
    st.success(f"💰 Predicted Salary: ${round(salary, 2)}")
