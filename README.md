# Real-Time Fraud Detection System

A production-style ML pipeline that scores credit card transactions
for fraud in real time using XGBoost, FastAPI, and Streamlit.

## Features
- XGBoost model trained on 284,000 real transactions
- FastAPI REST endpoint with <100ms inference
- Live Streamlit dashboard showing fraud scores as transactions stream
- SMOTE applied to handle extreme class imbalance (0.17% fraud rate)

## Tech Stack
Python, XGBoost, FastAPI, Streamlit, scikit-learn, imbalanced-learn

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python3 notebooks/train_model.py`
3. Start the API: `uvicorn api.main:app --reload`
4. Start the dashboard: `streamlit run app/dashboard.py`

## Results
- ROC-AUC: ~0.97
- Precision on fraud class: ~0.85
- Recall on fraud class: ~0.80

## Dataset
Credit Card Fraud Detection — Kaggle (mlg-ulb/creditcardfraud)
