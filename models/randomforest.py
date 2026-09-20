# ============================================
# RANDOM FOREST CLASSIFICATION MODEL
# Cosmetics Customer Dataset
# ============================================

# 1. Import libraries
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
    confusion_matrix,
    classification_report
)
from pathlib import Path
# ============================================
# 2. Load the cleaned dataset
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ============================================
# 3. Define features and target
# ============================================

# Target variable
y = df["purchased"]

# Features
X = df.drop("purchased", axis=1)

# ============================================
# 4. Remove ID column
# ============================================

# ID is only an identifier and should not be used
# as a meaningful predictive feature.
X = X.drop("Id", axis=1)

# ============================================
# 5. Identify numerical and categorical columns
# ============================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns

print("\nCategorical columns:")
print(list(categorical_columns))

print("\nNumerical columns:")
print(list(numerical_columns))

# ============================================
# 6. Preprocessing
# ============================================

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

# ============================================
# 7. Create Random Forest model
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

# ============================================
# 8. Create complete pipeline
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", rf_model)
    ]
)

# ============================================
# 9. Split dataset into training and testing
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ============================================
# 10. Train the Random Forest model
# ============================================

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# ============================================
# 11. Make predictions
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 12. Evaluate the model
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# CONFUSION MATRIX PLOT
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Random Forest - Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Add values inside the boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ============================================
# 13. Feature importance
# ============================================

# Get feature names after One-Hot Encoding
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get importance from Random Forest
importances = model.named_steps[
    "random_forest"
].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(feature_importance.head(15))

# ============================================
# FEATURE IMPORTANCE PLOT
# ============================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "random_forest"
].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

# Select top 15 features
top_features = feature_importance.tail(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 15 Feature Importances - Random Forest")

plt.tight_layout()
plt.show()