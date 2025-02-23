import os
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Initialize FastAPI
app = FastAPI()

# Load Hugging Face token securely from environment variables
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

print("Loaded token")

def load_model():
    """Load the ML model with exception handling."""
    try:
        print("Loading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="Shashankhmg/citybike-demnd-prediction",
            filename="RF.joblib",
            use_auth_token=HF_TOKEN
        )
        print(f"Model loaded successfully: {model_path}")
        return joblib.load(model_path)

    except Exception as e:
        print(f"Error loading model: {e}")
        return None  # Return None if model loading fails

# Load model (catch any failure)
model = load_model()

if model is None:
    print("Model failed to load. Check your Hugging Face token and internet connection.")

# Define input data structure
class InputData(BaseModel):
    start_station_id: float
    hour_of_day: int
    day_of_week: int
    weekend: int
    month: int
    rush_hour: int
    avg_rolling_7days: int
    avg_rolling_30days: int
    start_lat: float
    start_lng: float

@app.get("/")
def home():
    """Check if FastAPI is running and if the model loaded."""
    if model is None:
        return {"message": "FastAPI is running, but model failed to load. Check logs."}
    return {"message": "FastAPI is running and model is loaded!"}

@app.get("/predict")
def predict_get(
    start_station_id: int, hour_of_day: int, day_of_week: int, weekend: int,
    month: int, rush_hour: int, avg_rolling_7days: float, avg_rolling_30days: float,
    start_lat: float, start_lng: float
):
    """Handle GET request predictions with exception handling."""
    try:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        features = np.array([[start_station_id, hour_of_day, day_of_week, weekend,
                              month, rush_hour, avg_rolling_7days, avg_rolling_30days,
                              start_lat, start_lng]])

        prediction = model.predict(features)
        return {"predicted_demand": prediction.tolist()}

    except Exception as e:
        print(f"Error in GET /predict: {e}")
        return {"error": str(e)}

@app.post("/predict")
def predict_post(data: InputData):
    """Handle POST request predictions with exception handling."""
    try:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        features = np.array([[data.start_station_id, data.hour_of_day, data.day_of_week, data.weekend,
                              data.month, data.rush_hour, data.avg_rolling_7days, data.avg_rolling_30days,
                              data.start_lat, data.start_lng]])

        prediction = model.predict(features)
        return {"predicted_demand": prediction.tolist()}

    except Exception as e:
        print(f"Error in POST /predict: {e}")
        return {"error": str(e)}

print('hello')
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Ensure it's an integer
    print(f"Starting server on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
