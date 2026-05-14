from __future__ import annotations

import pandas as pd


def _as_number(value):
    return pd.to_numeric(value, errors="coerce")


def infer_customer_value(row: pd.Series) -> str:
    cltv = _as_number(row.get("CLTV"))
    monthly = _as_number(row.get("Monthly Charges"))
    total = _as_number(row.get("Total Charges"))

    if pd.notna(cltv) and cltv >= 4000:
        return "High Value"
    if pd.notna(monthly) and monthly >= 75:
        return "High Value"
    if pd.notna(total) and total >= 3000:
        return "High Value"
    return "Standard Value"


def recommend_action(risk_category: str, customer_value: str = "Standard Value") -> str:
    if risk_category == "High Risk" and customer_value == "High Value":
        return "Offer premium retention package with priority support and loyalty benefits."
    if risk_category == "High Risk":
        return "Trigger retention call, diagnose dissatisfaction, and offer targeted plan adjustment."
    if risk_category == "Medium Risk":
        return "Send discount, engagement campaign, or service usage education based on customer profile."
    return "Continue monitoring through regular lifecycle communication."


def add_recommendations(scored_df: pd.DataFrame) -> pd.DataFrame:
    data = scored_df.copy()
    data["Customer_Value"] = data.apply(infer_customer_value, axis=1)
    data["Retention_Recommendation"] = data.apply(
        lambda row: recommend_action(row["Risk_Category"], row["Customer_Value"]),
        axis=1,
    )
    return data
