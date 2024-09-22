from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

app = FastAPI()

# loading models
model = pickle.load(open('models\mymodel.pkl', 'rb'))
scaler = pickle.load(open('models\scaler.pkl', 'rb'))  

class StarData(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    absolute_magnitude: float
    star_type: int
    Star_color_blue: int
    Star_color_blue_white: int
    Star_color_orange: int
    Star_color_orange_red: int
    Star_color_pale_yellow_orange: int
    Star_color_red: int
    Star_color_white: int
    Star_color_white_yellow: int
    Star_color_whitish: int
    Star_color_yellow_white: int
    Star_color_yellowish: int
    Star_color_yellowish_white: int

# Create a prediction endpoint
@app.post("/predict/")
async def predict(data: StarData):
    # Convert the input data into a DataFrame
    input_data = {'Temperature (K)': [data.temperature],'Luminosity(L/Lo)':data.luminosity,'Radius(R/Ro)':data.radius,'Absolute magnitude(Mv)':data.absolute_magnitude,'Star type':data.star_type,
              'Star_color_blue':[data.Star_color_blue], 'Star_color_blue_white':data.Star_color_blue_white, 'Star_color_orange':data.Star_color_orange, 'Star_color_orange_red':data.Star_color_orange_red, 'Star_color_pale_yellow_orange':data.Star_color_pale_yellow_orange,
              'Star_color_red':[data.Star_color_red], 'Star_color_white':data.Star_color_white, 'Star_color_white_yellow':data.Star_color_white_yellow, 'Star_color_whitish':data.Star_color_whitish, 'Star_color_yellow_white':data.Star_color_yellow_white, 
              'Star_color_yellowish':[data.Star_color_yellowish], 'Star_color_yellowish_white':data.Star_color_yellowish_white }
    
    input_data = pd.DataFrame(input_data)
    
    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_data[input_data.columns[:5]])

    input_data.iloc[0, :5] = input_scaled.flatten()

    
    # Perform the prediction
    prediction = model.predict(input_data)
    return {"prediction": prediction}
