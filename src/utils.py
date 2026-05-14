from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset2_v2.xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"
LEGACY_MODEL_PATH = PROJECT_ROOT / "logistic_model.pkl"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

TARGET_COLUMN = "Churn Value"
CUSTOMER_ID_COLUMN = "CustomerID"

DROP_COLUMNS = [
    "CustomerID",
    "Country",
    "State",
    "Count",
    "Churn Label",
    "Churn Reason",
    "Churn Score",
    "Churn Category",
    "Customer Status",
    "Lat Long",
    "Latitude",
    "Longitude",
    "City",
    "Zip Code",
]

HIGH_VALUE_COLUMNS = ["CLTV", "Monthly Charges", "Total Charges"]
DEFAULT_THRESHOLDS = {"medium": 0.40, "high": 0.70}


def ensure_directories() -> None:
    for path in (MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    model_path = Path(path)
    if model_path.exists():
        return model_path
    if model_path == DEFAULT_MODEL_PATH and LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return model_path


def read_customer_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-specific type fixes without mutating the caller's frame."""
    data = df.copy()
    for column in ("Total Charges", "Monthly Charges", "CLTV", "Tenure Months"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df)
    drop_columns = [col for col in DROP_COLUMNS if col in data.columns]
    if TARGET_COLUMN in data.columns:
        drop_columns.append(TARGET_COLUMN)
    return data.drop(columns=drop_columns, errors="ignore")


def get_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found.")
    return df[TARGET_COLUMN].astype(int)


def get_customer_ids(df: pd.DataFrame) -> pd.Series:
    if CUSTOMER_ID_COLUMN in df.columns:
        return df[CUSTOMER_ID_COLUMN].astype(str)
    return pd.Series([f"CUST-{idx + 1:05d}" for idx in range(len(df))], name=CUSTOMER_ID_COLUMN)


def risk_category(probability: float, thresholds: dict[str, float] | None = None) -> str:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if probability >= thresholds["high"]:
        return "High Risk"
    if probability >= thresholds["medium"]:
        return "Medium Risk"
    return "Low Risk"


def probability_percent(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def safe_existing_columns(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]
