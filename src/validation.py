from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.predict import _infer_expected_columns
from src.utils import TARGET_COLUMN


@dataclass
class SchemaValidationResult:
    is_valid: bool
    required_columns: list[str]
    missing_columns: list[str]
    extra_columns: list[str]


def expected_input_columns(artifact: dict) -> list[str]:
    columns = artifact.get("feature_columns")
    if columns:
        return list(columns)
    return _infer_expected_columns(artifact["model"]) or []


def validate_scoring_schema(df: pd.DataFrame, artifact: dict) -> SchemaValidationResult:
    required_columns = expected_input_columns(artifact)
    uploaded_columns = [col for col in df.columns if col != TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in uploaded_columns]
    extra_columns = [col for col in uploaded_columns if col not in required_columns]
    return SchemaValidationResult(
        is_valid=len(missing_columns) == 0,
        required_columns=required_columns,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
    )


def schema_template(artifact: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=expected_input_columns(artifact))
