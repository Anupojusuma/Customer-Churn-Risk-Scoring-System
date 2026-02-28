import streamlit as st
import joblib
import pandas as pd

st.title("Customer Churn Risk Dashboard")

uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv", "xlsx"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    with open("logistic_model.pkl", "rb") as f:
        model = joblib.load(f)

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    columns_to_drop = [
        'CustomerID','Country','State','Count',
        'Churn Label','Churn Reason','CLTV',
        'Lat Long','City','Zip Code'
    ]

    df_features = df.drop(columns=['Churn Value'], errors='ignore')
    df_features = df_features.drop(columns=columns_to_drop)

    df['Churn_Probability'] = model.predict_proba(df_features)[:, 1]

    st.write("Top High-Risk Customers")
    st.dataframe(df.sort_values(by='Churn_Probability', ascending=False).head(20))
