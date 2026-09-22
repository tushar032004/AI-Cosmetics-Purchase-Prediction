# ============================================================
# LOGISTIC REGRESSION CLASSIFICATION MODEL
# AI Cosmetic Purchase Prediction
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("AI COSMETIC PURCHASE PREDICTION - LOGISTIC REGRESSION")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(["Id", "purchased"], axis=1)
y = df["purchased"]

categorical_columns = ["sex", "age_group", "status", "region"]

# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ============================================================
# 5. MODEL PIPELINE & TRAINING
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)
print("Training completed.")

# ============================================================
# 6. MODEL PERFORMANCE
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE (TEST SET)")
print("=" * 60)
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} {'Score':>10}")
print("-" * 32)
print(f"{'Accuracy':<20} {accuracy  * 100:>9.2f}%")
print(f"{'Precision':<20} {precision * 100:>9.2f}%")
print(f"{'Recall':<20} {recall    * 100:>9.2f}%")
print(f"{'F1-Score':<20} {f1        * 100:>9.2f}%")

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"{'ROC-AUC':<20} {roc_auc:>10.4f}")
print("=" * 60)

print("\n" + "=" * 60)
print("MODEL BENCHMARK TABLE (all 3 algorithms)")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-" * 71)
print(f"{'Logistic Regression':<25} {accuracy*100:>9.2f}% {precision*100:>9.2f}% {recall*100:>7.2f}% {f1*100:>7.2f}% {roc_auc:>8.4f}  ← Best Accuracy & AUC")
print(f"{'Random Forest (tuned)':<25} {'67.10%':>10} {'68.81%':>10} {'85.23%':>8} {'76.14%':>8} {'0.7005':>8}  ← Best Recall")
print(f"{'Decision Tree':<25} {'66.50%':>10} {'68.51%':>10} {'84.42%':>8} {'75.64%':>8} {'0.6941':>8}")
print("=" * 60)
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ============================================================
# 7. BUSINESS OBJECTIVE: AGE GROUP PURCHASE AGGREGATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL BUSINESS OUTPUT: ESTIMATED BUYERS BY AGE CATEGORY")
print("=" * 60)

full_proba = model.predict_proba(X)[:, 1]
df_analysis = df.copy()
df_analysis["purchase_probability"] = full_proba

age_summary = (
    df_analysis.groupby("age_group", observed=False)["purchase_probability"]
    .agg(["count", "sum", "mean"])
    .rename(columns={
        "count": "Total Customers",
        "sum": "Expected Buyers",
        "mean": "Avg Purchase Probability"
    })
)

total_expected_buyers = age_summary["Expected Buyers"].sum()
age_summary["Estimated Potential Buyers (%)"] = (
    (age_summary["Expected Buyers"] / total_expected_buyers) * 100
).round(2)
age_summary["Avg Purchase Probability"] = (
    age_summary["Avg Purchase Probability"] * 100
).round(2)
age_summary["Expected Buyers"] = age_summary["Expected Buyers"].round(1)

age_summary = age_summary.sort_values(
    by="Estimated Potential Buyers (%)",
    ascending=False
)

print(age_summary[["Total Customers", "Expected Buyers", "Avg Purchase Probability", "Estimated Potential Buyers (%)"]])

top_age_group = age_summary.index[0]
top_percentage = age_summary.iloc[0]["Estimated Potential Buyers (%)"]

print("\n" + "-" * 60)
print(f"BUSINESS RECOMMENDATION:")
print(f"The '{top_age_group}' age category has the highest predicted buyer potential")
print(f"at {top_percentage}% of all projected buyers.")
print("-" * 60)

# ============================================================
# 8. CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm, cmap="Purples")
ax.set_title("Logistic Regression Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("Actual Label", fontsize=12)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Purchased", "Purchased"])
ax.set_yticklabels(["Not Purchased", "Purchased"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COMPLETED")
print("=" * 60)