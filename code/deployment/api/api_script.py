# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("model/star_classification_model.h5")

# Define a FastAPI app
app = FastAPI()

# Input data model (based on your dataset input features)
class StarData(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    absolute_magnitude: float

# Welcome/root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Star Classification API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: StarData):
    # Prepare the data for prediction (scaling/transformations, if required)
    input_data = np.array([[data.temperature, data.luminosity, data.radius, data.absolute_magnitude]])

    # Perform prediction
    prediction = model.predict(input_data)
    
    # Convert the prediction to a human-readable class label
    predicted_class = np.argmax(prediction, axis=1)
    
    # You can map the predicted class to the specific star category, for example:
    star_classes = {
        0: "Red Dwarf",
        1: "Brown Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Supergiant",
        5: "Hypergiant"
    }
    
    return {"prediction": star_classes.get(predicted_class[0], "Unknown")}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is healthy!"}
