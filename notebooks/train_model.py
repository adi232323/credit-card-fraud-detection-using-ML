import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Time", "Class"])
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, (y_prob > 0.5).astype(int)))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved!")
