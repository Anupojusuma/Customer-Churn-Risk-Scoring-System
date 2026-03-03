import streamlit as st
import joblib
import pandas as pd
import openpyxl

st.title("Customer Churn Risk Dashboard")


def safe_read_excel(file):
    wb = openpyxl.load_workbook(file, data_only=True)
    ws = wb.active
    data = list(ws.values)

    columns = data[0]          #
    rows = data[1:]           

    df = pd.DataFrame(rows, columns=columns)
    return df



uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Read CSV
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    
   
    else:
        df = safe_read_excel(uploaded_file)

  
    with open("logistic_model.pkl", "rb") as f:
        model = joblib.load(f)

    # Convert Total Charges safely
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Drop unnecessary columns
    columns_to_drop = [
        'CustomerID','Country','State','Count',
        'Churn Label','Churn Reason','CLTV',
        'Lat Long','City','Zip Code'
    ]

    df_features = df.drop(columns=['Churn Value'], errors='ignore')
    df_features = df_features.drop(columns=columns_to_drop)

    # Predict churn probability
    df['Churn_Probability'] = model.predict_proba(df_features)[:, 1]

    st.write("Top High-Risk Customers")
    st.dataframe(df.sort_values(by='Churn_Probability', ascending=False).head(20))
