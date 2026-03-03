import streamlit as st
import joblib
import pandas as pd
import openpyxl
import tempfile

st.title("Customer Churn Risk Dashboard")

# ---------- SAFE EXCEL READER ----------
def safe_read_excel(uploaded):
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    # Now open with openpyxl safely (bypasses filter metadata bug)
    wb = openpyxl.load_workbook(temp_path, data_only=True)
    ws = wb.active
    data = list(ws.values)

    columns = data[0]        # header row
    rows = data[1:]          # remaining data rows

    df = pd.DataFrame(rows, columns=columns)
    return df
# ---------------------------------------


uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv", "xlsx"])

if uploaded_file is not None:

    # CSV read
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    # XLSX read (safe)
    else:
        df = safe_read_excel(uploaded_file)

    # Load model
    with open("logistic_model.pkl", "rb") as f:
        model = joblib.load(f)

    # Convert numeric column
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Columns to drop
    columns_to_drop = [
        'CustomerID','Country','State','Count',
        'Churn Label','Churn Reason','CLTV',
        'Lat Long','City','Zip Code'
    ]

    # Drop features
    df_features = df.drop(columns=['Churn Value'], errors='ignore')
    df_features = df_features.drop(columns=columns_to_drop)

    # Predict churn probability
    df['Churn_Probability'] = model.predict_proba(df_features)[:, 1]

    st.write("Top High-Risk Customers")
    st.dataframe(df.sort_values(by='Churn_Probability', ascending=False).head(20))
