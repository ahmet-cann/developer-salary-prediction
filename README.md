# 🚀 Predicting Software Developer Salaries — A Regression Approach
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-red.svg)](https://pandas.pydata.org/)

This project investigates the factors influencing software developer compensation globally, utilizing advanced regression techniques to provide data-driven insights.

---

## 📂 01. Dataset Overview
The project is powered by the official Stack Overflow 2024 Dataset. Due to its size, the raw data is excluded from the repository.

🔗 **Link to Dataset:** [Stack Overflow Annual Developer Survey 2024 (Kaggle)](https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024)

- **Total Responses:** 65,000+
- **Cleaned Sample Size:** 5,608 entries
- **Target Variable:** `ConvertedCompYearly` (Annual Salary in USD)

---

## 🛠️ 02. Technical Methodology
A robust pipeline was implemented to ensure model accuracy and data integrity:

| Step | Action |
| :--- | :--- |
| **Data Cleaning** | Dropped missing compensation values and handled non-numeric experience entries. |
| **Outlier Removal** | Excluded extreme salary outliers (> $1,000,000) to stabilize training. |
| **Feature Engineering** | Engineered `NumLanguages` to quantify technical depth. |
| **Encoding** | Applied `LabelEncoder` for categorical variables like Country, Education, and Role. |

---

## 📊 03. Exploratory Data Analysis (EDA)
We analyzed geographic trends, experience-salary correlations, and education impacts.

<p align="center">
  <img src="eda_results.png" width="800" title="Exploratory Data Analysis">
</p>

---

## 🤖 04. Machine Learning Models & Results
Three powerful regression algorithms were evaluated based on **RMSE** (Error) and **R²** (Predictive Power).

### Performance Metrics Table
| Model | RMSE (Error) 📉 | R² (Accuracy) 📈 |
| :--- | :--- | :--- |
| Linear Regression | $73,123 | 0.152 |
| Random Forest | $65,582 | 0.318 |
| **Gradient Boosting** | **$60,818** | **0.413** |

<p align="center">
  <img src="model_comparison.png" width="800" title="Model Comparison">
</p>

---

## 🎯 05. Key Insights (Feature Importance)
What determines a developer's salary? Our Random Forest analysis highlights:

1. **Country (39.3%)**: Geography is the #1 factor.
2. **Experience (23.3%)**: Seniority leads to exponential growth.
3. **Tech Stack (13.6%)**: Knowing more languages correlates with higher pay.

<p align="center">
  <img src="feature_importance.png" width="600" title="Feature Importance">
</p>

---

## 🎓 06. Conclusion
- **Gradient Boosting** is the superior model for this tabular data.
- Practical experience and geographic location far outweigh formal education credentials in the current market.
- Future work involves hyperparameter tuning (GridSearchCV) and building a live Flask/Streamlit web app.

---
**COM2502 Introduction to Data Science**  
**Group 31** | **May 2026**
