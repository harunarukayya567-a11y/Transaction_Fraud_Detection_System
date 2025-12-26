import pandas as pd
import joblib
import streamlit as st

model = joblib.load('IsFraud_model.pkl')
encoder = joblib.load('IsFraud_encoder.pkl')

TransactionAmount = st.number_input("Enter transaction Amount:")
TransactionType = st.selectbox("Select transaction type", ["transfer", "payment", "withdraw", "deposit"])
TransactionTime = st.number_input("Enter transaction time:")
Location = st.selectbox("Select location", ["kano", "lagos", "abuja", "kaduna", "port harcourt"])
DeviceType = st.selectbox("Select device type", ["mobile", "web", "ATM"])


if st.button("Detect Fraud"):
    sample_data = pd.DataFrame({
    "TransactionAmount":[TransactionAmount],
    "TransactionType": [TransactionType],
    "TransactionTime": [TransactionTime],
    "Location": [Location],
    "DeviceType": [DeviceType]
    })
    
    converted = encoder.transform(sample_data)
    prediction = model.predict(converted)

    if prediction[0] == 1:
        st.warning("Potential fraud detected!")
    else:
        st.success("Transaction seems legitimate.")



