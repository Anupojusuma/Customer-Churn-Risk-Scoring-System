import pickle
import pandas as pd

# Load saved model
with open("../models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load new dataset
df = pd.read_excel("../data/raw/dataset2_v2.xlsx")

# Remove target if exists
if 'Churn Value' in df.columns:
    df = df.drop(columns=['Churn Value'])

# Drop same unused columns
columns_to_drop = [
    'CustomerID',
    'Country',
    'State',
    'Count',
    'Churn Label',
    'Churn Reason',
    'CLTV',
    'Lat Long',
    'City',
    'Zip Code'
]
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())
df_features = df.drop(columns=columns_to_drop)

# Predict probability
df['Churn_Probability'] = model.predict_proba(df_features)[:, 1]

# Save output
df.to_csv("../reports/churn_scored_customers.csv", index=False)

print("Batch scoring completed")

def risk_level(p):
    if p > 0.7:
        return "High"
    elif p > 0.4:
        return "Medium"
    else:
        return "Low"

df['Risk_Level'] = df['Churn_Probability'].apply(risk_level)

df = df.sort_values(by='Churn_Probability', ascending=False)

high_risk_df = df[df['Risk_Level'] == "High"]