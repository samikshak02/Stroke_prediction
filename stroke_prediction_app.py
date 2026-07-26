import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("stroke_scaler.pkl")
columns = joblib.load("stroke_columns.pkl")

st.title("Stroke Prediction System")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type",
                         ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status",
                              ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Create DataFrame
data = pd.DataFrame({
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender_Male": [1 if gender == "Male" else 0],
    "ever_married_Yes": [1 if ever_married == "Yes" else 0],
    "work_type_Never_worked": [1 if work_type == "Never_worked" else 0],
    "work_type_Private": [1 if work_type == "Private" else 0],
    "work_type_Self-employed": [1 if work_type == "Self-employed" else 0],
    "work_type_children": [1 if work_type == "children" else 0],
    "Residence_type_Urban": [1 if Residence_type == "Urban" else 0],
    "smoking_status_formerly smoked": [1 if smoking_status == "formerly smoked" else 0],
    "smoking_status_never smoked": [1 if smoking_status == "never smoked" else 0],
    "smoking_status_smokes": [1 if smoking_status == "smokes" else 0],
})

# Match Training Columns
data = data.reindex(columns=columns, fill_value=0)

# Scale Input
data = scaler.transform(data)

if st.button("Predict"):
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Stroke Risk Detected")
    else:
        st.success("✅ No Stroke Risk Detected")