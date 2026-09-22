# ============================================================
# RANDOM FOREST CLASSIFICATION MODEL
# AI Cosmetic Purchase Prediction
# ============================================================
# GridSearch optimised: n_estimators=300, max_depth=8,
#   min_samples_split=5, min_samples_leaf=2, max_features='sqrt'
#   class_weight=None  →  Accuracy: 67.10%, F1: 76.14%, AUC: 0.7003
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
# 2. FEATURES & TARGET
# ============================================================

X = df.drop(["Id", "purchased"], axis=1)
y = df["purchased"]

# ============================================================
# 3. CATEGORICAL & NUMERICAL COLUMNS
# ============================================================

categorical_columns = ["sex", "age_group", "status", "region"]
numerical_columns   = ["tenure", "total", "income", "quantity"]

print("\nCategorical columns:", categorical_columns)
print("Numerical columns  :", numerical_columns)

# ============================================================
# 4. PREPROCESSING PIPELINE
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
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")

# ============================================================
# 6. RANDOM FOREST MODEL  (GridSearch optimised)
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,        # 300 trees – best from GridSearch
                max_depth=8,             # prevents overfitting
                min_samples_split=5,     # min samples to split a node
                min_samples_leaf=2,      # min samples in a leaf
                max_features="sqrt",     # √features per split (standard)
                class_weight=None,       # no balancing – maximises accuracy
                random_state=42
            )
        )
    ]
)

# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining Random Forest (300 trees)...")
model.fit(X_train, y_train)
print("Training completed.")

# ============================================================
# 8. PREDICTIONS
# ============================================================

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================================
# 9. METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy  : {accuracy  * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall    * 100:.2f}%")
print(f"F1-Score  : {f1        * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 10. AGE-GROUP DEMAND AGGREGATION
#     Expected Buyers = Σ P(purchased=1 | xᵢ)  for each age group
# ============================================================

df_eval = df.copy()
full_probas = model.predict_proba(X)[:, 1]
df_eval["purchase_probability"] = full_probas

age_agg = (
    df_eval.groupby("age_group", observed=False)["purchase_probability"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "Total Customers",
                     "sum":   "Expected Buyers",
                     "mean":  "Avg Probability"})
)
total_expected = age_agg["Expected Buyers"].sum()
age_agg["Buyer Share (%)"] = (
    (age_agg["Expected Buyers"] / total_expected) * 100
).round(2)
age_agg = age_agg.sort_values("Buyer Share (%)", ascending=False)

print("\n" + "=" * 60)
print("AGE-GROUP EXPECTED BUYER DISTRIBUTION")
print("=" * 60)
print(age_agg.to_string())
top_group = age_agg.index[0]
top_share = age_agg.iloc[0]["Buyer Share (%)"]
print(f"\n→ PRIMARY TARGET: {top_group} age category ({top_share:.1f}% of expected buyers)")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)
importances = model.named_steps["random_forest"].feature_importances_

imp_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 FEATURE IMPORTANCES")
print("=" * 60)
print(imp_df.head(10).to_string(index=False))

# ============================================================
# 12. VISUALISATIONS
# ============================================================

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im = axes[0].imshow(cm, cmap="Greens")
axes[0].set_title("Random Forest – Confusion Matrix", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted Label");  axes[0].set_ylabel("Actual Label")
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Not Purchased", "Purchased"])
axes[0].set_yticklabels(["Not Purchased", "Purchased"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=18, fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")

# --- Feature Importance ---
top10 = imp_df.head(10).sort_values("Importance", ascending=True)
axes[1].barh(top10["Feature"], top10["Importance"], color="#2ca02c")
axes[1].set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL COMPLETE")
print("=" * 60)