from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability import (  # noqa: E402
    build_shap_interpretation_table,
    clean_feature_name,
    compute_feature_importance,
    explain_business_drivers,
    local_prediction_explanation,
    shap_summary_plot,
)
from src.predict import build_prediction_report, load_model_artifact, score_customers  # noqa: E402
from src.utils import DEFAULT_MODEL_PATH, FEATURE_IMPORTANCE_PATH, read_customer_data, resolve_model_path  # noqa: E402
from src.utils import TARGET_COLUMN  # noqa: E402
from src.validation import schema_template, validate_scoring_schema  # noqa: E402


st.set_page_config(
    page_title="Customer Churn Risk Intelligence",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: 0;}
    .kpi-card {
        background: #171d26;
        border: 1px solid #2a3442;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 112px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
    }
    .kpi-label {color: #aab7c4; font-size: 0.86rem; margin-bottom: 8px;}
    .kpi-value {color: #f8fafc; font-size: 2rem; line-height: 1.1; font-weight: 750;}
    .kpi-help {color: #8ea0b3; font-size: 0.82rem; margin-top: 8px;}
    .risk-high {color: #ff7a70; font-weight: 700;}
    .risk-medium {color: #f6b54b; font-weight: 700;}
    .risk-low {color: #4fd18b; font-weight: 700;}
    .section-note {color: #aab7c4; font-size: 0.95rem;}
    .action-card {
        background: #10283f;
        border: 1px solid #1f4e78;
        border-radius: 8px;
        padding: 16px;
        color: #ecf6ff;
        min-height: 132px;
    }
    .plain-card {
        background: #151b24;
        border: 1px solid #2a3442;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .workflow-card {
        background: #111821;
        border: 1px solid #263241;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 118px;
    }
</style>
"""


def read_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


@st.cache_resource(show_spinner=False)
def cached_artifact(model_path: str):
    return load_model_artifact(model_path)


def render_kpis(scored: pd.DataFrame) -> None:
    total = len(scored)
    high = int((scored["Risk_Category"] == "High Risk").sum())
    medium = int((scored["Risk_Category"] == "Medium Risk").sum())
    avg_probability = scored["Churn_Probability"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, "Customers Scored", f"{total:,}", "Total records processed"),
        (col2, "High-Risk Customers", f"{high:,}", f"{(high / max(total, 1)) * 100:.1f}% of portfolio"),
        (col3, "Medium-Risk Customers", f"{medium:,}", "Needs engagement campaign"),
        (col4, "Average Churn Probability", f"{avg_probability:.1f}%", "Portfolio-level risk"),
    ]
    for col, label, value, help_text in cards:
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_charts(scored: pd.DataFrame) -> None:
    left, right = st.columns([1, 1])
    risk_counts = (
        scored["Risk_Category"]
        .value_counts()
        .rename_axis("Risk Category")
        .reset_index(name="Customers")
    )
    order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts["Risk Category"] = pd.Categorical(risk_counts["Risk Category"], categories=order, ordered=True)
    risk_counts = risk_counts.sort_values("Risk Category")

    with left:
        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Customers",
            color="Risk Category",
            color_discrete_map={
                "Low Risk": "#178C5F",
                "Medium Risk": "#D98C00",
                "High Risk": "#C43D32",
            },
            title="Customer Risk Distribution",
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=60, b=20))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            scored,
            x="Churn_Probability",
            nbins=25,
            color="Risk_Category",
            color_discrete_map={
                "Low Risk": "#178C5F",
                "Medium Risk": "#D98C00",
                "High Risk": "#C43D32",
            },
            title="Churn Probability Spread",
        )
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), xaxis_tickformat=".0%")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def render_ground_truth_metrics(scored: pd.DataFrame) -> None:
    if TARGET_COLUMN not in scored.columns:
        st.info(
            "Prediction-only mode: this file has no actual churn label, so accuracy, precision, recall, "
            "F1-score, and ROC-AUC cannot be calculated yet. Use these scores for prioritization, then "
            "validate them after actual churn outcomes are collected."
        )
        return

    y_true = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    valid_mask = y_true.notna()
    if valid_mask.sum() == 0:
        st.caption("Ground-truth column is present, but it does not contain valid numeric churn labels.")
        return

    y_true = y_true[valid_mask].astype(int)
    y_pred = scored.loc[valid_mask, "Predicted_Churn"].astype(int)
    y_prob = scored.loc[valid_mask, "Churn_Probability"]
    cols = st.columns(5)
    metrics = [
        ("Validation Accuracy", accuracy_score(y_true, y_pred)),
        ("Precision", precision_score(y_true, y_pred, zero_division=0)),
        ("Recall", recall_score(y_true, y_pred, zero_division=0)),
        ("F1 Score", f1_score(y_true, y_pred, zero_division=0)),
        ("ROC-AUC", roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else None),
    ]
    st.subheader("Ground-Truth Validation")
    for col, (label, value) in zip(cols, metrics):
        display = "N/A" if value is None else f"{value:.3f}"
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.45rem;">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_workflow(has_ground_truth: bool) -> None:
    st.subheader("How This Tool Works")
    col1, col2, col3 = st.columns(3)
    active_train = "color:#4fd18b;" if has_ground_truth else ""
    active_predict = "color:#4fd18b;"
    active_monitor = "color:#f6b54b;" if not has_ground_truth else "color:#4fd18b;"
    col1.markdown(
        f"""
        <div class="workflow-card">
            <div class="kpi-label">1. Learn from the Past</div>
            <div style="line-height:1.55; {active_train}">
            We use historical data (where we know who stayed and who left) to teach the system what churn looks like.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="workflow-card">
            <div class="kpi-label">2. Predict the Present</div>
            <div style="line-height:1.55; {active_predict}">
            We feed your current, active customers into the system to calculate their risk of leaving and recommend actions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="workflow-card">
            <div class="kpi-label">3. Track the Future</div>
            <div style="line-height:1.55; {active_monitor}">
            As time passes, we see who actually leaves. We use this new information to make the system smarter over time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance(model) -> None:
    importance = compute_feature_importance(model, top_n=15)
    if importance.empty and FEATURE_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(15)

    if importance.empty:
        st.info("Feature importance is unavailable for the selected model.")
        return

    plot_importance = importance.copy()
    plot_importance["Driver"] = plot_importance["feature"].apply(clean_feature_name)
    plot_importance = plot_importance.sort_values("importance")
    fig = px.bar(
        plot_importance,
        x="importance",
        y="Driver",
        orientation="h",
        title="Top Features Influencing Churn",
        color_discrete_sequence=["#2F6F73"],
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), yaxis_title="")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    for note in explain_business_drivers(importance):
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def render_priority_customer(scored: pd.DataFrame, model) -> None:
    st.subheader("Priority Customer Review")
    risk_filter = st.radio(
        "Queue",
        ["High Risk", "Medium Risk", "All Customers"],
        index=0,
        horizontal=True,
    )
    if risk_filter == "All Customers":
        queue = scored.copy()
    else:
        queue = scored[scored["Risk_Category"] == risk_filter].copy()
    if queue.empty:
        st.info("No customers found for the selected queue.")
        return

    queue = queue.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)
    rank = st.slider("Priority rank in selected queue", 1, len(queue), 1)
    row = queue.iloc[[rank - 1]]

    probability = row["Churn_Probability"].iloc[0]
    risk = row["Risk_Category"].iloc[0]
    recommendation = row["Retention_Recommendation"].iloc[0]
    customer_id = row["CustomerID"].iloc[0]

    col1, col2 = st.columns([0.9, 1.1])
    col1.markdown(
        f"""
        <div class="plain-card">
            <div class="kpi-label">Selected customer</div>
            <div class="kpi-value" style="font-size:1.25rem;">#{rank} - {customer_id}</div>
            <div class="kpi-help">Churn probability: {probability * 100:.1f}%</div>
            <div class="kpi-help">Risk segment: {risk}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="action-card">
            <div class="kpi-label">Recommended retention action</div>
            <div style="font-size:1rem; line-height:1.55;">{recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    local_explanation = local_prediction_explanation(model, row)
    if not local_explanation.empty:
        local_explanation = local_explanation.drop(columns=["model_importance"], errors="ignore")
        st.caption("Why this customer appears in the priority queue")
        st.dataframe(local_explanation, use_container_width=True, hide_index=True)


def render_shap(model, data: pd.DataFrame) -> None:
    st.subheader("Model Explanation with SHAP")
    st.markdown(
        """
        SHAP explains how the trained model made its predictions. It does **not** prove real-world causality.
        Read it as model-audit evidence: points on the **right** push model churn risk higher, points on the
        **left** reduce model churn risk, and color shows whether the feature value is high/present or low/absent.
        """
    )
    interpretation = build_shap_interpretation_table(model)
    if not interpretation.empty:
        st.dataframe(interpretation, use_container_width=True, hide_index=True)
    if st.button("Generate SHAP Summary Plot"):
        try:
            with st.spinner("Generating SHAP summary plot..."):
                fig = shap_summary_plot(model, data)
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.warning(f"SHAP visualization is unavailable: {exc}")


def render_waiting_state(artifact: dict) -> None:
    st.info("Upload a customer file from the sidebar to run the prediction pipeline.")
    st.subheader("How This Project Should Be Used")
    col1, col2 = st.columns(2)
    col1.markdown(
        """
        <div class="plain-card">
            <div class="kpi-label">Demo / publication mode</div>
            <div style="line-height:1.55;">
            Use the bundled telecom churn dataset to show preprocessing, prediction, explainability,
            segmentation, validation metrics, and retention recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="plain-card">
            <div class="kpi-label">New dataset mode</div>
            <div style="line-height:1.55;">
            A new dataset should be retrained first if its columns differ. The current trained model
            can score only files with the same customer-feature schema.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Advanced: view/download expected model input columns"):
        template = schema_template(artifact)
        st.caption("These are the columns expected by the currently selected trained model.")
        st.dataframe(template, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Empty Input Template CSV",
            template.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_input_template.csv",
            mime="text/csv",
        )


def validate_or_stop(data: pd.DataFrame, artifact: dict) -> None:
    validation = validate_scoring_schema(data, artifact)
    if validation.is_valid:
        if validation.extra_columns:
            with st.expander("Extra columns detected"):
                st.write(
                    "These columns are present in the uploaded file but are not required by the model. "
                    "They will be kept in reports but not used for prediction."
                )
                st.write(validation.extra_columns)
        return

    st.error("Uploaded dataset does not match the model input schema.")
    st.write("Missing required columns:")
    st.dataframe(pd.DataFrame({"missing_column": validation.missing_columns}), use_container_width=True, hide_index=True)
    st.write(
        "For a different dataset with different columns, retrain the model on that dataset first. "
        "A trained churn model cannot produce ground-truth reliable predictions on an unrelated schema."
    )
    template = schema_template(artifact)
    st.download_button(
        "Download Required Template CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_required_template.csv",
        mime="text/csv",
    )
    st.stop()


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Customer Churn Risk Intelligence")
    st.caption("Explainable churn scoring, retention segmentation, and business-ready analytics.")

    default_model_path = resolve_model_path(DEFAULT_MODEL_PATH)
    model_path = st.sidebar.text_input("Model artifact", value=str(default_model_path))
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV or Excel", type=["csv", "xlsx", "xls"])

    if not Path(model_path).exists():
        st.error("No trained model artifact found. Run `python -m src.train` first.")
        st.stop()

    artifact = cached_artifact(model_path)
    model = artifact["model"]

    if uploaded_file is not None:
        data = read_upload(uploaded_file)
        st.sidebar.success("Uploaded dataset loaded.")
    else:
        render_waiting_state(artifact)
        st.stop()

    validate_or_stop(data, artifact)
    scored = score_customers(data, model_path=model_path)
    has_ground_truth = TARGET_COLUMN in scored.columns and pd.to_numeric(scored[TARGET_COLUMN], errors="coerce").notna().any()

    render_kpis(scored)
    render_prediction_workflow(has_ground_truth)
    render_ground_truth_metrics(scored)
    st.divider()
    render_charts(scored)

    st.subheader("High-Risk Customer Queue")
    visible_columns = [
        "CustomerID",
        "Churn_Probability_%",
        "Risk_Category",
        "Customer_Value",
        "Retention_Recommendation",
    ]
    available = [col for col in visible_columns if col in scored.columns]
    st.dataframe(scored[available].head(30), use_container_width=True, hide_index=True)

    report = build_prediction_report(scored)
    st.download_button(
        "Download Prediction Report CSV",
        report.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_prediction_report.csv",
        mime="text/csv",
    )

    st.divider()
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        render_feature_importance(model)
    with col2:
        render_priority_customer(scored, model)

    st.divider()
    render_shap(model, data)


if __name__ == "__main__":
    main()
