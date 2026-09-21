from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

def load_data(filepath=None):
    """Load cleaned customer purchase dataset."""
    path = Path(filepath) if filepath else PROCESSED_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)

def get_features_and_target(df):
    """Separate features and target, removing identifier columns."""
    drop_cols = [col for col in ["Id", "purchased"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["purchased"] if "purchased" in df.columns else None
    return X, y
