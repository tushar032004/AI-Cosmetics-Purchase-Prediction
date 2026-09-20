
 # ============================================
# COSMETICS PRODUCT PURCHASE - DECISION TREE
# ============================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================
# 2. Load the dataset
# ============================================
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

df = pd.read_csv(CSV_PATH)

print(df.head())


print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================
# 3. Check the target variable
# ============================================

print("\nPurchase distribution:")
print(df["purchased"].value_counts())


# ============================================
# 4. Define features (X) and target (y)
# ============================================

# We don't use Id because it is only an identifier.
# We want to predict/train on 'purchased'.

X = df.drop(["Id", "purchased"], axis=1)

y = df["purchased"]


# ============================================
# 5. Define categorical columns
# ============================================

categorical_columns = [
    "sex",
    "age_group",
    "status",
    "region"
]


# ============================================
# 6. Preprocess categorical data
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
# 7. Split data into training and testing data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 8. Create Decision Tree model
# ============================================

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


# ============================================
# 9. TRAIN THE DECISION TREE
# ============================================

model.fit(X_train, y_train)

print("\nDecision Tree training completed!")


# ============================================
# 10. Visualise the trained Decision Tree
# ============================================

tree = model.named_steps["decision_tree"]

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree - Cosmetics Product Purchase")

plt.show()
