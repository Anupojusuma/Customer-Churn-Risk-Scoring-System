#  Customer Churn Risk Scoring System

##  Project Overview

This project builds an end-to-end Machine Learning system to predict customer churn using demographic, service usage, and billing data. 

The system identifies high-risk customers and assigns churn probability scores to support proactive retention strategies.

---

##  Business Problem

Customer churn directly impacts revenue in subscription-based businesses. 

Instead of randomly targeting customers, this system:
- Predicts churn probability for each customer
- Categorizes customers into risk levels
- Enables focused retention efforts

---

##  Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Logistic Regression
- Pipeline & ColumnTransformer

---

##  Project Structure
#  Customer Churn Risk Scoring System

##  Project Overview

This project builds an end-to-end Machine Learning system to predict customer churn using demographic, service usage, and billing data. 

The system identifies high-risk customers and assigns churn probability scores to support proactive retention strategies.

---

##  Business Problem

Customer churn directly impacts revenue in subscription-based businesses. 

Instead of randomly targeting customers, this system:
- Predicts churn probability for each customer
- Categorizes customers into risk levels
- Enables focused retention efforts

---

##  Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Logistic Regression
- Pipeline & ColumnTransformer

---

##  Project Structure
Customer_churn_prediction/
├── data/
│ └── raw/
│ └── dataset2_v2.xlsx
│
├── models/
│ └── logistic_model.pkl
│
├── reports/
│ └── churn_scored_customers.csv
│
├── src/
│ ├── train.ipynb
│ └── score.py
|
└── README.md


---

##  Model Development

### Data Processing
- Handled missing values
- Converted numeric columns appropriately
- Used OneHotEncoding for categorical variables
- Applied StandardScaler for numerical features

### Model Comparison
Two models were trained and evaluated:
- Logistic Regression
- Random Forest

Evaluation metrics:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score

Logistic Regression achieved higher ROC-AUC and was selected as the final production model.

---

##  Batch Risk Scoring

The `score.py` script:
- Loads the trained model (.pkl)
- Scores the entire dataset
- Generates churn probability
- Assigns risk levels (High / Medium / Low)
- Exports a scored dataset

---

##  Streamlit Dashboard

The project includes a simple dashboard built with Streamlit that allows:

- Uploading customer dataset
- Automatic churn scoring
- Viewing top high-risk customers
- Downloading scored dataset

Run locally using:
streamlit run app.py


---

##  Risk Categorization

| Probability | Risk Level |
|------------|------------|
| > 0.7      | High       |
| 0.4 – 0.7  | Medium     |
| < 0.4      | Low        |

---

##  Key Learnings

- End-to-end ML workflow (training → serialization → inference)
- Model comparison & selection
- Batch prediction system design
- Separation of training and inference stages
- Building a lightweight ML dashboard

---


