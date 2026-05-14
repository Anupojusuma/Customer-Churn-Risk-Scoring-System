from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.recommendation_engine import add_recommendations
from src.utils import (
    DEFAULT_MODEL_PATH,
    TARGET_COLUMN,
    get_customer_ids,
    get_feature_frame,
    normalize_columns,
    probability_percent,
    resolve_model_path,
    risk_category,
)


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = resolve_model_path(model_path)
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "thresholds": None, "feature_columns": None}


def _infer_expected_columns(model) -> list[str] | None:
    try:
        preprocessor = model.named_steps["preprocessor"]
        columns = []
        for _, _, transformer_columns in preprocessor.transformers_:
            if isinstance(transformer_columns, list):
                columns.extend(transformer_columns)
        return columns or None
    except Exception:
        return None


def _align_features(df: pd.DataFrame, expected_columns: list[str] | None) -> pd.DataFrame:
    if not expected_columns:
        return get_feature_frame(df)
    data = normalize_columns(df).drop(columns=[TARGET_COLUMN], errors="ignore")
    for column in expected_columns:
        if column not in data.columns:
            data[column] = pd.NA
    return data[expected_columns]


def score_customers(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    thresholds = thresholds or artifact.get("thresholds")

    expected_columns = artifact.get("feature_columns") or _infer_expected_columns(model)
    features = _align_features(df, expected_columns)
    probabilities = model.predict_proba(features)[:, 1]

    scored = df.copy()
    scored["CustomerID"] = get_customer_ids(df)
    scored["Churn_Probability"] = probabilities
    scored["Churn_Probability_%"] = [probability_percent(p) for p in probabilities]
    scored["Predicted_Churn"] = (probabilities >= 0.50).astype(int)
    scored["Risk_Category"] = [risk_category(p, thresholds) for p in probabilities]
    scored = add_recommendations(scored)
    return scored.sort_values("Churn_Probability", ascending=False)


def build_prediction_report(scored_df: pd.DataFrame) -> pd.DataFrame:
    report_columns = [
        "CustomerID",
        "Churn_Probability",
        "Churn_Probability_%",
        "Risk_Category",
        "Customer_Value",
        "Retention_Recommendation",
    ]
    available_columns = [col for col in report_columns if col in scored_df.columns]
    return scored_df[available_columns].copy()


def save_prediction_report(scored_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_prediction_report(scored_df).to_csv(output_path, index=False)
    return output_path
