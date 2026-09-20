# ============================================================
# LOGISTIC REGRESSION
# AI Cosmetic Purchase Prediction
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_cosmetics_dataset.csv"
)

df = pd.read_csv(CSV_PATH)


# ============================================================
# DISPLAY DATA INFORMATION
# ============================================================

print("=" * 60)
print("AI COSMETIC PURCHASE PREDICTION")
print("LOGISTIC REGRESSION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    ["Id", "purchased"],
    axis=1
)

y = df["purchased"]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "sex",
    "age_group",
    "status",
    "region"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# PRINT PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# MATPLOTLIB - CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

im = ax.imshow(cm)

ax.set_title(
    "Logistic Regression Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=12
)

ax.set_ylabel(
    "Actual Label",
    fontsize=12
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    [
        "Not Purchased",
        "Purchased"
    ]
)

ax.set_yticklabels(
    [
        "Not Purchased",
        "Purchased"
    ]
)

# Add values to matrix
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

plt.tight_layout()

plt.show()


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COMPLETED")
print("=" * 60)