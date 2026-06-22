# Predicting Software Developer Salaries Based on Skill Sets — A Regression Approach

## 01. Problem Statement & Motivation
Can we predict a software developer's salary based on their skills, experience, and background? This project explores geographic location, years of experience, and technical education to understand global salary disparities.

## 02. Dataset Overview
- **Source:** Stack Overflow Annual Developer Survey 2024
- **Clean Records:** 5,608 entries
- **Target Variable:** `ConvertedCompYearly` (Annual salary in USD)

## 03. Exploratory Data Analysis (EDA)
Insights into salary distribution, experience correlation, and the impact of education levels.

![EDA Results](eda_results.png)

## 04. Machine Learning Models
We applied three regression models to evaluate performance:
1. **Linear Regression:** Baseline model for linear relationships.
2. **Random Forest Regressor:** Ensemble method handling non-linearity (100 trees).
3. **Gradient Boosting Regressor:** Sequential tree building for maximum accuracy (Best Performer).

## 05. Results & Model Comparison
The models were evaluated using **RMSE** (Root Mean Squared Error) and **R²** (Coefficient of Determination).

![Model Comparison](model_comparison.png)

| Model | RMSE | R² |
| :--- | :--- | :--- |
| Linear Regression | $73,123 | 0.152 |
| Random Forest | $65,582 | 0.318 |
| **Gradient Boosting** | **$60,818** | **0.413** |

## 06. Feature Importance
What drives salaries the most? According to our Random Forest analysis:

![Feature Importance](feature_importance.png)

- **Country (39.3%):** Geographic location is the primary driver.
- **Experience (23.3%):** Years of coding significantly impact pay.
- **Num Languages (13.6%):** Broader tech stacks correlate with higher compensation.

## 07. Conclusion
- Gradient Boosting is our champion model with an R² of 0.413.
- Geographic location remains the most significant factor in compensation.
- Practical experience outweighs formal education in the current market.

---
**Course:** COM2502 — Introduction to Data Science  
**Group 31** | May 2026
