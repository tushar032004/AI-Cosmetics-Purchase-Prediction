from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CATEGORICAL_COLUMNS = ["sex", "age_group", "status", "region"]

def build_preprocessor():
    """
    Build ColumnTransformer for one-hot encoding categorical variables.
    Returns a NEW instance each call — never share across pipelines.
    """
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ],
        remainder="passthrough"
    )

def build_model_pipeline(model_type="logistic_regression"):
    """
    Instantiate a complete machine learning pipeline by model name.

    Args:
        model_type: one of 'decision_tree', 'logistic_regression', 'random_forest'

    Returns:
        sklearn Pipeline with preprocessor + classifier

    Model benchmarks (20% stratified test set, 10,000 rows):
        Logistic Regression : Acc=67.90%, F1=76.40%, AUC=0.7116  ← Best Accuracy & AUC
        Random Forest (opt) : Acc=67.10%, F1=76.14%, AUC=0.7005  ← Best Recall (85.23%)
        Decision Tree       : Acc=66.50%, F1=75.64%, AUC=0.6941
    """
    preprocessor = build_preprocessor()

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    elif model_type == "random_forest":
        # GridSearch optimised (320 candidates, 5-fold CV)
        # Best params: n_estimators=300, max_depth=8, min_samples_split=5,
        #              min_samples_leaf=2, max_features='sqrt', class_weight=None
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            random_state=42
        )
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            "Choose from: 'decision_tree', 'logistic_regression', 'random_forest'"
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
