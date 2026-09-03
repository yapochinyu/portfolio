from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


model = joblib.load("models/RandomForest_pipeline.pkl") 

@app.get("/")
def root():
    return {"message": "Heart Disease API is running"}

@app.post("/predict")

def predict(data: dict):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }