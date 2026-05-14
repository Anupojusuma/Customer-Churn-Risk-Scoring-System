# Customer Churn Prediction Analysis Report

## Objective

The objective of this project is to predict customer churn risk and support retention decision-making through explainable machine learning, customer segmentation, and recommendation logic.

## Dataset

The system uses a telecom customer dataset containing demographic attributes, service subscriptions, contract information, billing behavior, customer lifetime value, and churn labels.

## Methodology

The project uses a reusable preprocessing pipeline that handles missing values, converts numeric billing fields, encodes categorical variables, and scales numerical features. The training workflow compares Logistic Regression, Random Forest, and XGBoost when available. Class imbalance is addressed through class-weight balancing for scikit-learn models.

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Cross-validation ROC-AUC

## Explainability

The framework includes global feature importance and SHAP summary plots. Global importance identifies the strongest model drivers across the portfolio, while the individual customer panel provides a local driver snapshot to help business users interpret a selected prediction. SHAP values explain model behavior and should not be interpreted as proof of real-world causality.

## Labeled and Unlabeled Data Handling

The system distinguishes between labeled historical datasets and unlabeled current customer datasets. If the uploaded file contains the actual `Churn Value`, the dashboard reports ground-truth validation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. If the uploaded file does not contain `Churn Value`, the system produces prediction-only outputs: churn probability, risk category, and retention recommendation. In prediction-only mode, performance metrics are unavailable until actual churn outcomes are collected later.

## Business Impact

The output is designed for retention teams. Each customer receives a churn probability, a risk category, a customer value segment, and a recommended retention action. This turns model output into an operational queue that can guide targeted outreach. After the retention period, actual outcomes can be compared with predicted risk scores to monitor model usefulness and guide retraining.

## Generated Artifacts

- `models/best_churn_model.joblib`
- `reports/model_metrics.csv`
- `reports/confusion_matrix.csv`
- `reports/feature_importance.csv`
- `reports/churn_scored_customers.csv`
- `reports/prediction_report.csv`
