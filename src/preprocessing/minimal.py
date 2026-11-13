"""
Minimal preprocessing (Part A, step 1): data/raw/cars_dataframe.csv -> data/processed/cars_minimal.csv

This is the baseline the full preprocessing is compared against, so it deliberately does as little as possible:
  - Price (the target) missing  -> remove the row (a made-up price would be fake data)
  - Numeric feature missing     -> 0 (a constant: nothing is learned from the data, so it is not imputation)
  - Text feature missing        -> remove the row
  - Every text column stays as text; it is one-hot encoded later in src/shared/core_utils.py
  - Price stays in MAD (no log transform, no outlier removal, no feature selection)
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_minimal.csv"
TARGET = "Price"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Target: remove rows without a price
    df = df.dropna(subset=[TARGET]).copy()

    # Numeric features (only "Number of Doors" in the raw data): missing -> 0
    numeric = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df[numeric] = df[numeric].fillna(0)

    # Text features: remove rows with a missing value
    text = [c for c in df.columns if c not in numeric + [TARGET]]
    df = df.dropna(subset=text)
    return df


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df_clean = preprocess(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"[Minimal preprocessing] Rows: {len(df):,} -> {len(df_clean):,}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
