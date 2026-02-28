from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease using a Random Forest model trained on the UCI Cleveland dataset.",
    version="1.0.0"
)

# ── Input / Output schemas ──────────────────────────────────────────────────

class HeartData(BaseModel):
    age: float
    sex: float           # 1 = male, 0 = female
    cp: float            # chest pain type (0-3)
    trestbps: float      # resting blood pressure
    chol: float          # serum cholesterol in mg/dl
    fbs: float           # fasting blood sugar > 120 mg/dl (1=true)
    restecg: float       # resting ECG results (0-2)
    thalach: float       # max heart rate achieved
    exang: float         # exercise induced angina (1=yes)
    oldpeak: float       # ST depression induced by exercise
    slope: float         # slope of peak exercise ST segment
    ca: float            # number of major vessels (0-3)
    thal: float          # thal: 3=normal, 6=fixed defect, 7=reversable defect

class HeartResponse(BaseModel):
    prediction: int
    label: str
    confidence: float

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running. Visit /docs for usage."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "dataset": "UCI Cleveland Heart Disease",
        "n_estimators": 100,
        "max_depth": 6,
        "features": 13,
        "target": "Binary: 0 = No Disease, 1 = Disease"
    }

@app.post("/predict", response_model=HeartResponse)
async def predict_heart_disease(data: HeartData):
    try:
        features = [
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]
        result = predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))