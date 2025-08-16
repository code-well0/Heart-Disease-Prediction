import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)

sex_val = 1 if sex == "Male" else 0
features = np.array([[age, sex_val, cp, trestbps, chol]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("Heart Disease" if prediction == 1 else "No Heart Disease")
