import streamlit as st
import requests, pandas as pd, time, random

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Real-Time Fraud Detection Monitor")

df_raw = pd.read_csv("data/creditcard.csv").sample(200, random_state=42)

log = []
placeholder = st.empty()

st.write("Streaming 200 transactions...")

for _, row in df_raw.iterrows():
    transaction = row.drop(["Time", "Class"]).to_dict()
    try:
        res = requests.post(
            "http://localhost:8000/predict",
            json=transaction,
            timeout=2
        )
        result = res.json()
        result["actual"] = int(row["Class"])
        log.append(result)

        df_log = pd.DataFrame(log)
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Transactions processed", len(df_log))
            col2.metric("Flagged (BLOCK/REVIEW)",
                        len(df_log[df_log.decision != "APPROVE"]))
            col3.metric("Avg fraud probability",
                        f"{df_log.fraud_probability.mean():.4f}")
            st.line_chart(df_log["fraud_probability"])
            st.dataframe(df_log.tail(10))
    except:
        pass
    time.sleep(0.05)

st.success("Done streaming!")
