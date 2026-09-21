# ============================================================
# DECISION TREE CLASSIFICATION MODEL
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
print("AI COSMETIC PURCHASE PREDICTION - DECISION TREE")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================

# Exclude Id (identifier) and target column
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
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

print("\nTraining Decision Tree...")
model.fit(X_train, y_train)
print("Decision Tree training completed!")

# ============================================================
# 6. MODEL EVALUATION
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
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ============================================================
# 7. BUSINESS OBJECTIVE: AGE GROUP PURCHASE AGGREGATION
# ============================================================
# AI predicts purchase probability; business analysis turns
# these predictions into marketing insights by summing probabilities.

print("\n" + "=" * 60)
print("FINAL BUSINESS OUTPUT: ESTIMATED BUYERS BY AGE CATEGORY")
print("=" * 60)

# Estimate on full dataset to give company-wide forecast
full_proba = model.predict_proba(X)[:, 1]
df_analysis = df.copy()
df_analysis["purchase_probability"] = full_proba

# Sum predicted probabilities per age group (expected number of buyers)
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

# Sort by potential share
age_summary = age_summary.sort_values(
    by="Estimated Potential Buyers (%)",
    ascending=False
)

print(age_summary[["Total Customers", "Expected Buyers", "Avg Purchase Probability", "Estimated Potential Buyers (%)"]])

top_age_group = age_summary.index[0]
top_percentage = age_summary.iloc[0]["Estimated Potential Buyers (%)"]

print("\n" + "-" * 60)
print(f"BUSINESS RECOMMENDATION:")
print(f"The '{top_age_group}' age category represents the largest expected buyer potential")
print(f"({top_percentage}% of all anticipated purchases). Marketing campaigns and")
print(f"product launch budgets should prioritize this demographic segment.")
print("-" * 60)

# ============================================================
# 8. VISUALISATIONS
# ============================================================

# 1. Decision Tree Visualization
tree = model.named_steps["decision_tree"]
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    max_depth=3,
    fontsize=10,
    rounded=True
)
plt.title("Decision Tree (First 3 Levels) - Cosmetics Purchase Prediction")
plt.tight_layout()
plt.show()

# 2. Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_title("Decision Tree - Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Purchased", "Purchased"])
ax.set_yticklabels(["Not Purchased", "Purchased"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()
