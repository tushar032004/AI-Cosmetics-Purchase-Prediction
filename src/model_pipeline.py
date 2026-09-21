from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CATEGORICAL_COLUMNS = ["sex", "age_group", "status", "region"]

def build_preprocessor():
    """Build ColumnTransformer for one-hot encoding categorical variables."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ],
        remainder="passthrough"
    )

def build_model_pipeline(model_type="random_forest"):
    """Instantiate a complete machine learning pipeline by model name."""
    preprocessor = build_preprocessor()
    
    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
