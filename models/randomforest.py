# ============================================================
# RANDOM FOREST CLASSIFICATION MODEL
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
from sklearn.ensemble import RandomForestClassifier
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
print("AI COSMETIC PURCHASE PREDICTION - RANDOM FOREST")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
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
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ============================================================
# 5. RANDOM FOREST MODEL PIPELINE
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", rf_model)
    ]
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

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
print("MODEL EVALUATION (TEST SET)")
print("=" * 60)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

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
print(f"The '{top_age_group}' age category has the highest estimated potential buyers")
print(f"({top_percentage}%). Allocate the majority of the launch budget to this group.")
print("-" * 60)

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["random_forest"].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCES")
print("=" * 60)
print(feature_importance.head(10))

# ============================================================
# 9. VISUALISATIONS
# ============================================================

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontweight="bold")

plt.tight_layout()
plt.show()

# Feature Importance Plot
top_features = feature_importance.tail(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"], color="seagreen")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()