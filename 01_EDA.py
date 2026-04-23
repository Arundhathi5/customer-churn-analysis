# ============================================================
# Customer Churn Analysis - 01: Exploratory Data Analysis
# Author: Arundhathi Reddy
# GitHub: https://github.com/Arundhathi5
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 5)

# ============================================================
# 1. LOAD DATA
# ============================================================
# Using the IBM Telco Customer Churn dataset (public)
# Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place the CSV in the ../data/ folder

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nFirst 5 rows:\n", df.head())

# ============================================================
# 2. DATA CLEANING
# ============================================================

# TotalCharges has spaces instead of NaN — fix it
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Encode target variable
df['Churn_Flag'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"\nChurn rate: {df['Churn_Flag'].mean():.2%}")
print(f"Dataset size after cleaning: {df.shape[0]:,} rows")

# ============================================================
# 3. CHURN DISTRIBUTION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
churn_counts = df['Churn'].value_counts()
axes[0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

# Pie chart
axes[1].pie(churn_counts.values, labels=churn_counts.index,
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Rate', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. CONTRACT TYPE vs CHURN
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
contract_churn = df.groupby('Contract')['Churn_Flag'].mean().reset_index()
bars = ax.bar(contract_churn['Contract'], contract_churn['Churn_Flag'] * 100,
              color=['#e74c3c', '#f39c12', '#2ecc71'])
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Contract Type')
for bar, val in zip(bars, contract_churn['Churn_Flag']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/02_contract_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. TENURE vs CHURN
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df[df['Churn'] == 'No']['tenure'], bins=30, alpha=0.6,
             color='#2ecc71', label='No Churn')
axes[0].hist(df[df['Churn'] == 'Yes']['tenure'], bins=30, alpha=0.6,
             color='#e74c3c', label='Churned')
axes[0].set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Count')
axes[0].legend()

# Box plot
df.boxplot(column='tenure', by='Churn', ax=axes[1],
           boxprops=dict(color='#2c3e50'),
           medianprops=dict(color='#e74c3c', linewidth=2))
axes[1].set_title('Tenure by Churn Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Tenure (months)')

plt.tight_layout()
plt.savefig('../reports/03_tenure_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. MONTHLY CHARGES vs CHURN
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
for churn_val, color, label in [('No', '#2ecc71', 'No Churn'), ('Yes', '#e74c3c', 'Churned')]:
    subset = df[df['Churn'] == churn_val]['MonthlyCharges']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=label)
ax.set_title('Monthly Charges Distribution by Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/04_monthly_charges_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. CORRELATION HEATMAP (Numeric Features)
# ============================================================

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Flag']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. KEY EDA FINDINGS SUMMARY
# ============================================================

print("\n" + "="*55)
print("KEY EDA FINDINGS")
print("="*55)
print(f"Overall churn rate       : {df['Churn_Flag'].mean():.2%}")
print(f"Avg tenure (churned)     : {df[df['Churn']==' Yes']['tenure'].mean():.1f} months")
print(f"Avg tenure (retained)    : {df[df['Churn']=='No']['tenure'].mean():.1f} months")
print(f"Avg monthly charge churn : ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}")
print(f"Avg monthly charge retain: ${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}")
print(f"Month-to-month churn rate: {df[df['Contract']=='Month-to-month']['Churn_Flag'].mean():.2%}")
print(f"Two-year contract churn  : {df[df['Contract']=='Two year']['Churn_Flag'].mean():.2%}")
print("="*55)
