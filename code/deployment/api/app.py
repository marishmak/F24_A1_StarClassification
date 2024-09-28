from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from fastapi.responses import JSONResponse

app = FastAPI()

# loading models
model = pickle.load(open(os.path.join('models', 'content', 'mymodel.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join('models', 'content', 'scaler.pkl'), 'rb'))

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

# Input data schema using Pydantic
class StarData(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    absolute_magnitude: float
    star_type: float
    Star_color_blue: float
    Star_color_blue_white: float
    Star_color_orange: float
    Star_color_orange_red: float
    Star_color_pale_yellow_orange: float
    Star_color_red: float
    Star_color_white: float
    Star_color_white_yellow: float
    Star_color_whitish: float
    Star_color_yellow_white: float
    Star_color_yellowish: float
    Star_color_yellowish_white: float

# Create a prediction endpoint
@app.post("/predict")
async def predict(data: StarData):
    try:
        # Correct feature names to match those used during model training
        input_data = pd.DataFrame([{
            'Temperature (K)': data.temperature,
            'Luminosity(L/Lo)': data.luminosity,
            'Radius(R/Ro)': data.radius,
            'Absolute magnitude(Mv)': data.absolute_magnitude,
            'Star type': data.star_type,
            'Star color_blue': data.Star_color_blue,
            'Star color_blue white': data.Star_color_blue_white,
            'Star color_orange': data.Star_color_orange,
            'Star color_orange red': data.Star_color_orange_red,
            'Star color_pale yellow orange': data.Star_color_pale_yellow_orange,
            'Star color_red': data.Star_color_red,
            'Star color_white': data.Star_color_white,
            'Star color_white yellow': data.Star_color_white_yellow,
            'Star color_whitish': data.Star_color_whitish,
            'Star color_yellow white': data.Star_color_yellow_white,
            'Star color_yellowish': data.Star_color_yellowish,
            'Star color_yellowish white': data.Star_color_yellowish_white
        }])

        # Scale the input data using the same scaler used during training
        scaled_data = scaler.transform(input_data)

        # Perform the prediction
        prediction = model.predict(scaled_data)

        # Ensure that the prediction is returned as is, without forcing integer conversion
        return {"prediction": str(prediction[0])}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
