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
print(f"Dataset shape : {df.shape}")
print(f"Target balance: {df['purchased'].value_counts().to_dict()}")
print(df.head())

# ============================================================
# 2. FEATURES & TARGET
# ============================================================

X = df.drop(["Id", "purchased"], axis=1)
y = df["purchased"]

# ============================================================
# 3. PREPROCESSING
# ============================================================

categorical_columns = ["sex", "age_group", "status", "region"]
numerical_columns   = ["tenure", "total", "income", "quantity"]

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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")

# ============================================================
# 5. DECISION TREE MODEL  (max_depth=5 prevents overfitting)
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

# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining Decision Tree (max_depth=5)...")
model.fit(X_train, y_train)
print("Training completed.")

# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================================
# 8. METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} {'Score':>10}")
print("-" * 32)
print(f"{'Accuracy':<20} {accuracy  * 100:>9.2f}%")
print(f"{'Precision':<20} {precision * 100:>9.2f}%")
print(f"{'Recall':<20} {recall    * 100:>9.2f}%")
print(f"{'F1-Score':<20} {f1        * 100:>9.2f}%")
print(f"{'ROC-AUC':<20} {roc_auc:>10.4f}")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================
# 9. MODEL COMPARISON CONTEXT
# ============================================================

print("\n" + "=" * 60)
print("MODEL BENCHMARK TABLE (all 3 algorithms)")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-" * 71)
print(f"{'Logistic Regression':<25} {'67.90%':>10} {'69.83%':>10} {'84.33%':>8} {'76.40%':>8} {'0.7116':>8}  ← Best Accuracy")
print(f"{'Random Forest (tuned)':<25} {'67.10%':>10} {'68.81%':>10} {'85.23%':>8} {'76.14%':>8} {'0.7005':>8}  ← Best Recall")
print(f"{'Decision Tree':<25} {accuracy*100:>9.2f}% {precision*100:>9.2f}% {recall*100:>7.2f}% {f1*100:>7.2f}% {roc_auc:>8.4f}")
print("=" * 60)

# ============================================================
# 10. AGE-GROUP DEMAND AGGREGATION
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
# 11. VISUALISATIONS
# ============================================================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
trained_tree  = model.named_steps["decision_tree"]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Decision Tree (first 3 levels) ---
plot_tree(
    trained_tree,
    feature_names=feature_names,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    max_depth=3,
    fontsize=9,
    rounded=True,
    ax=axes[0]
)
axes[0].set_title("Decision Tree – First 3 Levels", fontsize=13, fontweight="bold")

# --- Confusion Matrix ---
axes[1].imshow(cm, cmap="Purples")
axes[1].set_title("Decision Tree – Confusion Matrix", fontsize=13, fontweight="bold")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Not Purchased", "Purchased"])
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Not Purchased", "Purchased"])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=16, fontweight="bold")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("DECISION TREE MODEL COMPLETE")
print("=" * 60)
