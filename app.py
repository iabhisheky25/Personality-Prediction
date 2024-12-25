import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('personality_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Personality Prediction API")

st.write("Provide the following details:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
openness = st.number_input("Openness (1-10)", min_value=1.0, max_value=10.0, value=5.0)
neuroticism = st.number_input("Neuroticism (1-10)", min_value=1.0, max_value=10.0, value=5.0)
conscientiousness = st.number_input("Conscientiousness (1-10)", min_value=1.0, max_value=10.0, value=5.0)
agreeableness = st.number_input("Agreeableness (1-10)", min_value=1.0, max_value=10.0, value=5.0)
extraversion = st.number_input("Extraversion (1-10)", min_value=1.0, max_value=10.0, value=5.0)
gender = st.radio("Gender", ("Male", "Female"))

# Convert gender to numerical value (0 for Male, 1 for Female)
gender_numeric = 0 if gender == "Male" else 1

# Prediction button
if st.button("Predict Personality"):
    # Prepare the input data
    input_data = [age, openness, neuroticism, conscientiousness, agreeableness, extraversion, gender_numeric]
    input_data_df = pd.DataFrame([input_data], columns=['age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion', 'gender'])

    # Scale the input data
    scaled_input_data = scaler.transform(input_data_df)

    # Predict the personality trait
    prediction = model.predict(scaled_input_data)

    # Display the result
    st.write(f"Predicted Personality Trait: {prediction[0]}")
