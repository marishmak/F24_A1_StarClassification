# app.py
import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle


# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# loading models
model = pickle.load(open('models\mymodel.pkl', 'rb'))
scaler = pickle.load(open('models\scaler.pkl', 'rb'))  

# Title and instructions for the app
st.title("Star Classification Prediction App")
st.write("""
This app predicts the star class based on input features.""")

# Input fields for the star parameters
temperature = st.number_input("Temperature (K)")
luminosity = st.number_input("Luminosity (L/Lo)")
radius = st.number_input("Radius (R/Ro)")
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)")
star_type = st.number_input("Star type")
Star_color_blue = st.number_input("Is star blue")
Star_color_blue_white = st.number_input("Is star blue-white")
Star_color_orange = st.number_input("Is star orange")
Star_color_orange_red = st.number_input("Is star orange-red")
Star_color_pale_yellow_orange = st.number_input("Is star pale yellow-orange")
Star_color_red = st.number_input("Is star red")
Star_color_white = st.number_input("Is star white")
Star_color_white_yellow = st.number_input("Is star white-yellow")
Star_color_whitish = st.number_input("Is star whitish")
Star_color_yellow_white = st.number_input("Is star yellow-white")
Star_color_yellowish = st.number_input("Is star yellowish")
Star_color_yellowish_white = st.number_input("Is star yellowish-white")

# Button to trigger prediction
if st.button("Classify the Star"):
    # Prepare input data for prediction
    input_data = np.array([[temperature, luminosity, radius, absolute_magnitude, star_type,
                            Star_color_blue, Star_color_blue_white, Star_color_orange,
                            Star_color_orange_red, Star_color_pale_yellow_orange,
                            Star_color_red, Star_color_white, Star_color_white_yellow,
                            Star_color_whitish, Star_color_yellow_white, Star_color_yellowish,
                            Star_color_yellowish_white]])
    
    
    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_data[input_data.columns[:5]])

    input_data.iloc[0, :5] = input_scaled.flatten()

    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()["prediction"]

    st.success(f"The star is classified as: {prediction}")

# Footer information
st.write("Developed by Mariia Shmakova")
