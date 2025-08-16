import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

st.title("❤️ Heart Disease Prediction App")

# Input fields (exactly 11 features)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
max_hr = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", [0, 1, 2])

# Convert categorical inputs if needed
sex_num = 1 if sex == "Male" else 0

# Prepare input for model (11 features only)
features = np.array([[
    age,
    sex_num,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope
]])

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ The model predicts a risk of heart disease.")
    else:
        st.success("✅ The model predicts no heart disease risk.")
