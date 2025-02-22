from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
from huggingface_hub import hf_hub_download
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load Hugging Face token securely from environment variables
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")
print('loaded token')
def load_model():
    print("Loading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="Shashankhmg/citybike-demnd-prediction", 
        filename="RF.joblib",
        use_auth_token=HF_TOKEN
    )
    print("Model loaded successfully:", model_path)
    return joblib.load(model_path)

model = load_model()

# Define input data structure
class InputData(BaseModel):
    start_station_id: int
    hour_of_day: int
    day_of_week: int
    weekend: int
    month: int
    rush_hour: int
    avg_rolling_7days: float
    avg_rolling_30days: float
    start_lat: float
    start_lng: float

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# GET request for /predict (example values for testing in browser)
@app.get("/predict")
def predict_get(
    start_station_id: int, hour_of_day: int, day_of_week: int, weekend: int, 
    month: int, rush_hour: int, avg_rolling_7days: float, avg_rolling_30days: float, 
    start_lat: float, start_lng: float
):
    features = np.array([[start_station_id, hour_of_day, day_of_week, weekend, 
                          month, rush_hour, avg_rolling_7days, avg_rolling_30days, 
                          start_lat, start_lng]])
    
    prediction = model.predict(features)
    return {"predicted_demand": prediction.tolist()}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.start_station_id, data.hour_of_day, data.day_of_week, data.weekend, 
                          data.month, data.rush_hour, data.avg_rolling_7days, data.avg_rolling_30days, 
                          data.start_lat, data.start_lng]])
    
    prediction = model.predict(features)
    return {"predicted_demand": prediction.tolist()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Default to 8080
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
