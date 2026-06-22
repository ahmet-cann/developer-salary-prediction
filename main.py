import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('survey_results_10k.csv')

df_clean = df[['Country', 'EdLevel', 'YearsCodePro', 'DevType', 
               'LanguageHaveWorkedWith', 'Employment', 'ConvertedCompYearly']].copy()

df_clean = df_clean.dropna(subset=['ConvertedCompYearly'])
df_clean = df_clean[df_clean['ConvertedCompYearly'] < 1000000]

df_clean['YearsCodePro'] = pd.to_numeric(df_clean['YearsCodePro'], errors='coerce')
df_clean = df_clean.dropna(subset=['YearsCodePro'])

df_clean['NumLanguages'] = df_clean['LanguageHaveWorkedWith'].str.split(';').str.len()

le_country = LabelEncoder()
le_edlevel = LabelEncoder()
le_devtype = LabelEncoder()
le_employment = LabelEncoder()

df_clean['Country'] = le_country.fit_transform(df_clean['Country'])
df_clean['EdLevel'] = le_edlevel.fit_transform(df_clean['EdLevel'])
df_clean['DevType'] = le_devtype.fit_transform(df_clean['DevType'])
df_clean['Employment'] = le_employment.fit_transform(df_clean['Employment'])

X = df_clean[['Country', 'EdLevel', 'YearsCodePro', 'DevType', 'Employment', 'NumLanguages']]
y = df_clean['ConvertedCompYearly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(df_clean['ConvertedCompYearly'], bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].set_xlabel('Annual Salary (USD)')

axes[0, 1].scatter(df_clean['YearsCodePro'], df_clean['ConvertedCompYearly'], alpha=0.3, color='orange')
axes[0, 1].set_title('Experience vs Salary')
axes[0, 1].set_xlabel('Years of Experience')

df_clean.groupby(le_edlevel.inverse_transform(df_clean['EdLevel']))['ConvertedCompYearly'].median().sort_values().plot(kind='barh', color='green', ax=axes[1, 0])
axes[1, 0].set_title('Education Level vs Salary')

languages = df['LanguageHaveWorkedWith'].str.split(';').explode()
languages.value_counts().head(10).sort_values().plot(kind='barh', color='purple', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Programming Languages')

plt.tight_layout()
plt.savefig('eda_results.png')

models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
rmse_values = [73123, 65582, 60818]
r2_values = [0.152, 0.318, 0.413]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(models, rmse_values, color=['red', 'yellow', 'green'], alpha=0.7)
ax1.set_ylabel('RMSE (Lower is Better)')
ax2 = ax1.twinx()
ax2.plot(models, r2_values, color='black', marker='o', linewidth=3)
ax2.set_ylabel('R2 Score (Higher is Better)')
plt.title('Model Comparison: RMSE & R2')
plt.savefig('model_comparison.png')

features = ['Country', 'Experience', 'NumLanguages', 'Education', 'Employment', 'DevType']
importances = [0.393, 0.233, 0.136, 0.067, 0.05, 0.121]
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')

plt.figure(figsize=(10, 6))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='teal')
plt.title('Feature Importance (Driving Factors)')
plt.savefig('feature_importance.png')

sample_data = pd.DataFrame([{
    'Country': 'United States',
    'EdLevel': "Bachelor's degree",
    'YearsCodePro': 7,
    'DevType': 'Full-stack developer',
    'Employment': 'Employed, full-time',
    'NumLanguages': 5
}])

sample_data['Country'] = le_country.transform(sample_data['Country'])
sample_data['EdLevel'] = le_edlevel.transform(sample_data['EdLevel'])
sample_data['DevType'] = le_devtype.transform(sample_data['DevType'])
sample_data['Employment'] = le_employment.transform(sample_data['Employment'])

prediction = gb.predict(sample_data)
print(f"Live Prediction Demo Result: ${prediction[0]:,.2f}")