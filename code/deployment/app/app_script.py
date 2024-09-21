# app.py
import streamlit as st
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("model/star_classification_model.h5")

# Title and instructions for the app
st.title("Star Classification Prediction App")
st.write("""
This app predicts the star class based on input features such as 
**Temperature**, **Luminosity**, **Radius**, and **Absolute Magnitude**.
""")

# Input fields for the star parameters
temperature = st.number_input("Temperature (K)", min_value=0.0, value=5000.0)
luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0, value=1.0)
radius = st.number_input("Radius (R/Ro)", min_value=0.0, value=1.0)
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", value=4.83)

# Button to trigger prediction
if st.button("Classify Star"):
    # Prepare input data for prediction
    input_data = np.array([[temperature, luminosity, radius, absolute_magnitude]])

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Get the predicted class (highest probability)
    predicted_class = np.argmax(prediction, axis=1)

    # Map the predicted class to the star category
    star_classes = {
        0: "Red Dwarf",
        1: "Brown Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Supergiant",
        5: "Hypergiant"
    }

    st.success(f"The star is classified as: {star_classes.get(predicted_class[0], 'Unknown')}")

# Footer information
st.write("Developed by [Your Name]")
