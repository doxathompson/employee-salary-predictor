from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load artifacts
model = joblib.load("app/model.pkl")
scaler = joblib.load("app/scaler.pkl")
features = joblib.load("app/features.pkl")

# Define the input data model
class EmployeeData(BaseModel):
    Age: float
    Gender: str
    Education_Level: str
    Job_Title: str
    Years_of_Experience: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Salary Predictor API!"}



# Prediction route
@app.post("/predict")
def predict_salary(data: EmployeeData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Education Level": data.Education_Level,
        "Job Title": data.Job_Title,
        "Years of Experience": data.Years_of_Experience
    }])

    # One-hot encode to match training
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Scale numeric features
    input_encoded[["Age", "Years of Experience"]] = scaler.transform(
        input_encoded[["Age", "Years of Experience"]])

    # Predict
    prediction = model.predict(input_encoded)[0]
    return {"predicted_salary": round(prediction, 2)}
