# customer-churn-analysis
# 📉 Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-336791?logo=postgresql&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?logo=tableau&logoColor=white)

## 📌 Project Overview

A machine learning and analytics project that analyzes 50,000+ customer records to identify behavioral patterns driving churn. Using a Random Forest classifier with 87% accuracy, combined with EDA and a Tableau dashboard, this project delivers actionable retention strategy recommendations.

---

## 🚀 Key Features

- **Analyzed 50K+ customer records** using Python and SQL to identify churn patterns
- **Random Forest classifier** achieving **87% model accuracy**
- **Feature engineering & EDA** to identify top churn drivers: contract type, tenure, and monthly charges
- **Tableau dashboard** visualizing churn segments by region, product tier, and tenure
- **Stakeholder-ready report** with actionable retention recommendations
- **30% reduction in data prep time** via optimized SQL queries

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Analysis | Python (Pandas, NumPy) |
| Machine Learning | Scikit-learn (Random Forest, Feature Importance) |
| Visualization | Seaborn, Matplotlib, Tableau |
| Data Storage | SQL (PostgreSQL) |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 87% |
| Model | Random Forest Classifier |
| Dataset Size | 50,000+ records |
| Key Features | Contract Type, Tenure, Monthly Charges |

---

## 🔍 Key Insights

- **Contract type** is the strongest predictor of churn — month-to-month customers churn at 3x the rate of annual contract holders
- **Tenure** inversely correlates with churn — customers in the first 12 months are highest risk
- **Monthly charges** above a certain threshold significantly increase churn probability

---

## 📁 Project Structure

```
customer-churn-analysis/
├── data/
│   └── README.md              # Data description (raw data not included)
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── sql/
│   └── customer_data_prep.sql # Data extraction & aggregation
├── dashboards/
│   └── churn_dashboard.twbx   # Tableau workbook
├── reports/
│   └── churn_analysis_report.pdf
└── README.md
```

---

## 📈 Sample Code

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")  # 87%
```

---

