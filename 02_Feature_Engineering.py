# ============================================================
# Customer Churn Analysis - 02: Feature Engineering
# Author: Arundhathi Reddy
# GitHub: https://github.com/Arundhathi5
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop customerID — not a feature
df.drop(columns=['customerID'], inplace=True)

print("Data loaded:", df.shape)

# ============================================================
# 2. ENCODE BINARY COLUMNS
# ============================================================

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']

for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

print("Binary encoding done.")

# ============================================================
# 3. ENCODE MULTI-CLASS COLUMNS (One-Hot)
# ============================================================

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract',
              'PaymentMethod']

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("One-hot encoding done. Shape:", df.shape)

# ============================================================
# 4. FEATURE ENGINEERING — NEW FEATURES
# ============================================================

# Avg monthly spend relative to tenure
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

# Tenure buckets
df['TenureBucket'] = pd.cut(df['tenure'],
                             bins=[0, 12, 24, 48, 72],
                             labels=['0-12mo', '13-24mo', '25-48mo', '49+mo'])
df = pd.get_dummies(df, columns=['TenureBucket'], drop_first=True)

# High spender flag (above median monthly charges)
median_charge = df['MonthlyCharges'].median()
df['HighSpender'] = (df['MonthlyCharges'] > median_charge).astype(int)

print("Feature engineering done. New shape:", df.shape)

# ============================================================
# 5. FEATURE IMPORTANCE PREVIEW (Correlation with Churn)
# ============================================================

corr_with_churn = df.corr()['Churn'].drop('Churn').sort_values(key=abs, ascending=False)

print("\nTop 15 features correlated with Churn:")
print(corr_with_churn.head(15).to_string())

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
top_features = corr_with_churn.head(15)
colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_features.values]
ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top 15 Features Correlated with Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation with Churn')
plt.tight_layout()
plt.savefig('../reports/06_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. SCALE NUMERIC FEATURES
# ============================================================

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("\nFeatures scaled.")

# ============================================================
# 7. SAVE PROCESSED DATASET
# ============================================================

df.to_csv('../data/churn_processed.csv', index=False)
print(f"\nProcessed dataset saved: {df.shape[0]:,} rows, {df.shape[1]} features")
print("Target distribution:\n", df['Churn'].value_counts())
