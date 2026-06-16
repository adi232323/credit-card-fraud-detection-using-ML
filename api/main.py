from fastapi import FastAPI
import joblib, pandas as pd

app = FastAPI()
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    df["Amount"] = scaler.transform(df[["Amount"]])
    prob = model.predict_proba(df)[0][1]
    decision = "BLOCK" if prob > 0.5 else (
               "REVIEW" if prob > 0.3 else "APPROVE")
    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision
    }
