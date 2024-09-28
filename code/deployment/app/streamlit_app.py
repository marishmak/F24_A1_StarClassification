# app.py
import streamlit as st
import requests


# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Title and instructions for the app
st.title("Star Classification Prediction App")
st.write("""
This app predicts the star class based on input features.""")

# Input fields for the star parameters
temperature = st.number_input("Temperature (K)", min_value=1000.0)
luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0)
radius = st.number_input("Radius (R/Ro)", min_value=0.0)
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", min_value=0.0)
star_type = st.number_input("Star type", min_value=0.0)
Star_color_blue = st.number_input("Is star blue", min_value=0.0)
Star_color_blue_white = st.number_input("Is star blue-white", min_value=0.0)
Star_color_orange = st.number_input("Is star orange", min_value=0.0)
Star_color_orange_red = st.number_input("Is star orange-red", min_value=0.0)
Star_color_pale_yellow_orange = st.number_input("Is star pale yellow-orange", min_value=0.0)
Star_color_red = st.number_input("Is star red", min_value=0.0)
Star_color_white = st.number_input("Is star white", min_value=0.0)
Star_color_white_yellow = st.number_input("Is star white-yellow", min_value=0.0)
Star_color_whitish = st.number_input("Is star whitish", min_value=0.0)
Star_color_yellow_white = st.number_input("Is star yellow-white", min_value=0.0)
Star_color_yellowish = st.number_input("Is star yellowish", min_value=0.0)
Star_color_yellowish_white = st.number_input("Is star yellowish-white", min_value=0.0)

# Button to trigger prediction
if st.button("Classify the Star"):
    # Prepare input data for prediction
#     input_data = {
#     "temperature": 5778.0,
#     "luminosity": 1.0,
#     "radius": 1.0,
#     "absolute_magnitude": 4.83,
#     "star_type": 1,
#     "Star_color_blue": 0.0,
#     "Star_color_blue_white": 0.0,
#     "Star_color_orange": 0.0,
#     "Star_color_orange_red": 0.0,
#     "Star_color_pale_yellow_orange": 0.0,
#     "Star_color_red": 0.0,
#     "Star_color_white": 1.0,
#     "Star_color_white_yellow": 0.0,
#     "Star_color_whitish": 0.0,
#     "Star_color_yellow_white": 1.0,
#     "Star_color_yellowish": 0.0,
#     "Star_color_yellowish_white": 0.0
# }

    
    input_data = {'temperature': temperature,'luminosity':luminosity,'radius':radius,'absolute_magnitude':absolute_magnitude,'star_type':star_type,
              'Star_color_blue':Star_color_blue, 'Star_color_blue_white':Star_color_blue_white, 'Star_color_orange':Star_color_orange, 'Star_color_orange_red':Star_color_orange_red, 'Star_color_pale_yellow_orange':Star_color_pale_yellow_orange,
              'Star_color_red':Star_color_red, 'Star_color_white':Star_color_white, 'Star_color_white_yellow':Star_color_white_yellow, 'Star_color_whitish':Star_color_whitish, 'Star_color_yellow_white':Star_color_yellow_white, 
              'Star_color_yellowish':Star_color_yellowish, 'Star_color_yellowish_white':Star_color_yellowish_white }

    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()['prediction']

    st.success(f"The star is classified as: {prediction}")

# Footer information
st.write("Developed by Mariia Shmakova")
