import pickle
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "heart_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(features: list) -> dict:
    model = load_model()
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0]
    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    return {
        "prediction": int(pred),
        "label": label,
        "confidence": round(float(max(prob)), 4)
    }