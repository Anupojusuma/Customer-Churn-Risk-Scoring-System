from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.explainability import save_feature_importance
from src.preprocessing import build_preprocessor, infer_feature_types
from src.utils import (
    DATA_PATH,
    DEFAULT_MODEL_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    DEFAULT_THRESHOLDS,
    ensure_directories,
    get_feature_frame,
    get_target,
    read_customer_data,
)


RANDOM_STATE = 42


def build_candidate_models(preprocessor) -> dict[str, tuple[Pipeline, dict]]:
    candidates: dict[str, tuple[Pipeline, dict]] = {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {"classifier__C": [0.3, 1.0, 3.0]},
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 8, 14],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_estimators=250,
                            learning_rate=0.05,
                            max_depth=4,
                            subsample=0.9,
                            colsample_bytree=0.9,
                        ),
                    ),
                ]
            ),
            {
                "classifier__max_depth": [3, 4],
                "classifier__learning_rate": [0.03, 0.05],
            },
        )
    except Exception:
        pass

    return candidates


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_models(data_path: str | Path = DATA_PATH, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    ensure_directories()
    df = read_customer_data(data_path)
    X = get_feature_frame(df)
    y = get_target(df)

    numerical_features, categorical_features = infer_feature_types(X)
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidate_models(preprocessor)

    records = []
    fitted_models = {}
    for name, (pipeline, param_grid) in candidates.items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)
        cv_scores = cross_val_score(best_model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        records.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
                "best_params": search.best_params_,
                **metrics,
            }
        )
        fitted_models[name] = best_model

    metrics_df = pd.DataFrame(records).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = best_model.predict(X_test)
    pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=["Actual_No_Churn", "Actual_Churn"],
        columns=["Predicted_No_Churn", "Predicted_Churn"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")

    pd.DataFrame({"y_true": y_test, "churn_probability": probabilities}).to_csv(
        REPORTS_DIR / "test_predictions.csv",
        index=False,
    )
    save_feature_importance(best_model)

    artifact = {
        "model": best_model,
        "best_model_name": best_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "thresholds": DEFAULT_THRESHOLDS,
        "feature_columns": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
    }
    joblib.dump(artifact, model_path)
    joblib.dump(best_model, Path(model_path).with_name("legacy_pipeline.joblib"))
    return artifact


def main() -> None:
    artifact = train_models()
    best = artifact["best_model_name"]
    print(f"Training complete. Best model: {best}")
    print(f"Saved model artifact to {DEFAULT_MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
