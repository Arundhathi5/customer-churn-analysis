# ============================================================
# Customer Churn Analysis - 03: Model Training & Evaluation
# Author: Arundhathi Reddy
# GitHub: https://github.com/Arundhathi5
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

df = pd.read_csv('../data/churn_processed.csv')

X = df.drop(columns=['Churn'])
y = df['Churn']

print(f"Features: {X.shape[1]}")
print(f"Samples : {X.shape[0]:,}")
print(f"Churn rate: {y.mean():.2%}")

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]:,}")
print(f"Test size : {X_test.shape[0]:,}")

# ============================================================
# 3. BASELINE RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n── Baseline Results ─────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churned']))

# ============================================================
# 4. HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='roc_auc', verbose=1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

print("\n── Tuned Model Results ──────────────────────────────")
print(f"Best params: {grid_search.best_params_}")
print(f"Accuracy   : {accuracy_score(y_test, y_pred_best):.2%}")
print(f"ROC-AUC    : {roc_auc_score(y_test, y_prob_best):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churned']))

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churned'],
            yticklabels=['No Churn', 'Churned'])
ax.set_title('Confusion Matrix — Tuned Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('../reports/07_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc_score = roc_auc_score(y_test, y_prob_best)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Customer Churn Model', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../reports/08_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top20 = importances.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20.index, top20.values, color='#3498db')
ax.set_title('Top 20 Feature Importances — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../reports/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 features driving churn:")
print(importances.nlargest(10).to_string())

# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "="*55)
print("MODEL SUMMARY")
print("="*55)
print(f"Model        : Random Forest Classifier")
print(f"Accuracy     : {accuracy_score(y_test, y_pred_best):.2%}")
print(f"ROC-AUC      : {roc_auc_score(y_test, y_prob_best):.4f}")
print(f"CV Accuracy  : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print(f"Top 3 drivers: {', '.join(importances.nlargest(3).index.tolist())}")
print("="*55)
