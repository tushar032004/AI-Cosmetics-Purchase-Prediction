# ============================================================
# DATA CLEANING & PREPROCESSING PIPELINE
# AI Cosmetic Purchase Prediction
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = BASE_DIR / "data" / "raw" / "cosmetics_kaggle_style_synthetic.csv"
OUTPUT_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

print("=" * 60)
print("COSMETICS DATA CLEANING PIPELINE")
print("=" * 60)

# 1. Load Raw Data
print(f"Loading raw dataset from: {RAW_CSV_PATH}")
df_raw = pd.read_csv(RAW_CSV_PATH)
print(f"Raw shape: {df_raw.shape}")
print("\nMissing values in raw data:")
print(df_raw.isnull().sum())

# 2. Check and Remove Duplicates
duplicates_count = df_raw.duplicated(subset=["Id"]).sum()
print(f"\nDuplicate customer IDs: {duplicates_count}")
df = df_raw.drop_duplicates(subset=["Id"]).copy()

# 3. Handle Missing Values
# - Impute numerical features with median
numeric_cols = ["tenure", "total", "income", "quantity"]
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed missing {col} with median: {median_val}")

# - Impute categorical features with mode
cat_cols = ["sex", "age_group", "status", "region"]
for col in cat_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed missing {col} with mode: {mode_val}")

# 4. Target variable validation
df = df.dropna(subset=["purchased"])
df["purchased"] = df["purchased"].astype(int)

# 5. Outlier Treatment & Logical Validation
# Ensure no negative spend, tenure, or income
df["total"] = df["total"].clip(lower=0)
df["tenure"] = df["tenure"].clip(lower=0)
df["income"] = df["income"].clip(lower=10000)
df["quantity"] = df["quantity"].clip(lower=1)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(f"\nCleaned dataset shape: {df.shape}")

# 6. Save Cleaned Dataset
OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nCleaned dataset successfully saved to: {OUTPUT_CSV_PATH}")
print("=" * 60)
