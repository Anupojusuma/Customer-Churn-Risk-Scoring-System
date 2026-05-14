from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FEATURE_IMPORTANCE_PATH, TARGET_COLUMN, get_feature_frame, normalize_columns


def clean_feature_name(feature: str) -> str:
    return feature.replace("_", " ").replace("  ", " ").strip()


def business_interpretation(feature: str) -> str:
    label = clean_feature_name(feature)
    lowered = feature.lower()
    if "month-to-month" in lowered:
        return "Month-to-month contract is usually a strong churn-risk signal. Prioritize contract renewal or loyalty offers."
    if "tenure" in lowered:
        return "Shorter tenure often indicates weaker customer stickiness. Use onboarding and early-life engagement."
    if "fiber optic" in lowered:
        return "Fiber optic customers may be sensitive to price or service quality. Check plan fit and support history."
    if "online security" in lowered or "tech support" in lowered:
        return "Lack of support/security add-ons can indicate lower product attachment. Promote relevant service bundles."
    if "monthly charges" in lowered or "total charges" in lowered:
        return "Billing level is influencing risk. Review affordability, discounts, and plan optimization."
    if "payment method" in lowered:
        return "Payment behavior is associated with risk. Encourage stable payment methods or billing reminders."
    if "dependents" in lowered or "partner" in lowered:
        return "Household profile affects churn tendency. Personalize campaign messaging by customer context."
    return f"{label} is contributing to model decisions and should be reviewed with customer history."


def get_feature_names(model) -> list[str]:
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        return []
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass

    names: list[str] = []
    try:
        for _, transformer, columns in preprocessor.transformers_:
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                names.extend(list(columns))
                continue

            final_step = transformer
            if hasattr(transformer, "steps"):
                final_step = transformer.steps[-1][1]

            if hasattr(final_step, "get_feature_names_out"):
                try:
                    names.extend(final_step.get_feature_names_out(columns).tolist())
                except TypeError:
                    names.extend(final_step.get_feature_names_out().tolist())
            else:
                names.extend(list(columns))
    except Exception:
        return []
    return names


def _get_classifier(model):
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]
        return list(model.named_steps.values())[-1]
    return model


def _expected_training_columns(model) -> list[str] | None:
    try:
        preprocessor = model.named_steps["preprocessor"]
        columns = []
        for _, _, transformer_columns in preprocessor.transformers_:
            if isinstance(transformer_columns, list):
                columns.extend(transformer_columns)
        return columns or None
    except Exception:
        return None


def _model_feature_frame(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = _expected_training_columns(model)
    if not expected_columns:
        return get_feature_frame(raw_df)
    data = normalize_columns(raw_df).drop(columns=[TARGET_COLUMN], errors="ignore")
    for column in expected_columns:
        if column not in data.columns:
            data[column] = pd.NA
    return data[expected_columns]


def compute_feature_importance(model, top_n: int = 20) -> pd.DataFrame:
    feature_names = get_feature_names(model)
    classifier = _get_classifier(model)
    if classifier is None or not feature_names:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(feature_names) != len(values):
        return pd.DataFrame(columns=["feature", "importance"])

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance


def save_feature_importance(model, output_path: str | Path = FEATURE_IMPORTANCE_PATH) -> pd.DataFrame:
    importance = compute_feature_importance(model, top_n=30)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def plot_feature_importance(importance: pd.DataFrame, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_data = importance.sort_values("importance", ascending=True)
    ax.barh(plot_data["feature"], plot_data["importance"], color="#2F6F73")
    ax.set_title("Top Churn Drivers")
    ax.set_xlabel("Model Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def shap_summary_plot(model, raw_df: pd.DataFrame, output_path: str | Path | None = None, max_rows: int = 300):
    try:
        import shap
    except Exception as exc:
        raise ImportError("SHAP is not installed. Install requirements.txt to enable SHAP plots.") from exc

    features = _model_feature_frame(model, raw_df).head(max_rows)
    preprocessor = model.named_steps["preprocessor"]
    classifier = _get_classifier(model)
    transformed = preprocessor.transform(features)
    feature_names = get_feature_names(model)
    if not feature_names or len(feature_names) != transformed.shape[1]:
        feature_names = [f"model_feature_{idx}" for idx in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(classifier, transformed_df)
        shap_values = explainer.shap_values(transformed_df)

    plt.figure(figsize=(9, 5.8))
    shap.summary_plot(
        shap_values,
        transformed_df,
        max_display=10,
        plot_size=(9, 5.8),
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def local_prediction_explanation(model, row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    feature_importance = compute_feature_importance(model, top_n=100)
    if feature_importance.empty:
        return feature_importance

    raw_features = get_feature_frame(row)
    available = []
    for feature in feature_importance["feature"]:
        matching_columns = [col for col in raw_features.columns if feature == col or feature.startswith(f"{col}_")]
        if matching_columns:
            source_col = max(matching_columns, key=len)
            value = raw_features.iloc[0][source_col]
            if feature.startswith(f"{source_col}_"):
                encoded_value = feature[len(source_col) + 1 :]
                value = f"{source_col} = {value}"
                if str(raw_features.iloc[0][source_col]) != encoded_value:
                    continue
            available.append(
                {
                    "Driver": clean_feature_name(feature),
                    "Customer signal": str(value),
                    "Business reading": business_interpretation(feature),
                    "model_importance": feature_importance.loc[
                        feature_importance["feature"] == feature, "importance"
                    ].iloc[0],
                }
            )
    return pd.DataFrame(available).head(top_n)


def explain_business_drivers(importance: pd.DataFrame) -> list[str]:
    if importance.empty:
        return ["Model-specific driver importances were not available for the selected estimator."]
    drivers = [clean_feature_name(feature) for feature in importance.head(5)["feature"].tolist()]
    return [
        "These are portfolio-level churn drivers. They show where retention teams should focus analysis first.",
        f"Current top drivers include: {', '.join(drivers)}.",
        "Use them as decision support, then combine with service history before selecting an intervention.",
    ]


def build_shap_interpretation_table(model, top_n: int = 6) -> pd.DataFrame:
    importance = compute_feature_importance(model, top_n=top_n)
    if importance.empty:
        return pd.DataFrame(columns=["Driver", "Business interpretation"])
    return pd.DataFrame(
        {
            "Driver": [clean_feature_name(feature) for feature in importance["feature"]],
            "Business interpretation": [business_interpretation(feature) for feature in importance["feature"]],
        }
    )
