# 🚀 CitiBike Demand Prediction API (FastAPI + Railway)

This repository contains a **FastAPI-based API** for CitiBike demand prediction, deployed on **Railway.app**. The API serves predictions using a **pre-trained Random Forest model** hosted on **Hugging Face**.

## Live API Endpoint
[FastAPI on Railway](https://fastapi-example-production-fdcd.up.railway.app/)

## Project Overview
- **Framework:** FastAPI
- **Hosting Platform:** Railway
- **Model Type:** Random Forest
- **Model Hosting:** [Hugging Face Model Repository](https://huggingface.co/Shashankhmg/citybike-demnd-prediction)
- **Prediction Task:** Estimates bike demand at CitiBike stations based on historical data.


## API Features
### 1. Load Pre-trained Model**
The model is **fetched from Hugging Face** using `hf_hub_download` and stored locally for reuse.

### 2. API Endpoints
#### Root Endpoint
- **`GET /`**
- Returns: `{ "message": "FastAPI is running!" }`

#### 3. Predict Demand (GET)
- **`GET /predict`**
- Parameters (query string):
  - `start_station_id` (float)
  - `hour_of_day` (int)
  - `day_of_week` (int)
  - `weekend` (int)
  - `month` (int)
  - `rush_hour` (int)
  - `avg_rolling_7days` (float)
  - `avg_rolling_30days` (float)
  - `start_lat` (float)
  - `start_lng` (float)
- Example Request:
  ```bash
  curl "https://fastapi-example-production-fdcd.up.railway.app/predict?start_station_id=6234.08&hour_of_day=10&day_of_week=2&weekend=0&month=6&rush_hour=1&avg_rolling_7days=50.3&avg_rolling_30days=48.2&start_lat=40.7128&start_lng=-74.0060"
  ```
- Example Response:
  ```json
  {
    "predicted_demand": [71.0]
  }
  ```

#### 3. Predict Demand (POST)
- **`POST /predict`**
- Body (JSON format):
  ```json
  {
    "start_station_id": 6234.08,
    "hour_of_day": 10,
    "day_of_week": 2,
    "weekend": 0,
    "month": 6,
    "rush_hour": 1,
    "avg_rolling_7days": 50.3,
    "avg_rolling_30days": 48.2,
    "start_lat": 40.7128,
    "start_lng": -74.0060
  }
  ```
- Example Response:
  ```json
  {
    "predicted_demand": [71.0]
  }
  ```

---

## ⚙️ Installation & Running Locally
### Prerequisites
- Python 3.8+
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/FastAPI-CitiBike.git
cd FastAPI-CitiBike
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Hugging Face Access Token
Create a `.env` file (or set as environment variables):
```bash
HF_ACCESS_TOKEN=your_huggingface_token
```

### 4. Run the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### 5. Test API Locally
```bash
curl "http://127.0.0.1:8080/"
```
