"""
Minimal preprocessing (Part A, step 1): data/raw/cars_dataframe.csv -> data/processed/cars_minimal.csv

The baseline uses the SAME ROWS as the full preprocessing (the rows full.py keeps, in the same order), so both
stages are trained and tested on the same cars and the comparison only measures what the column preprocessing adds.
The rows are chosen by full.preprocess itself; its cleaned values are thrown away.

For the columns it deliberately does as little as possible:
  - Numeric feature missing -> 0 (a constant: nothing is learned from the data, so it is not imputation)
  - Text feature missing    -> the text "missing" (its own one-hot category, so no row is lost)
  - Every raw column is kept and stays as it is (no car-name cleaning, no mileage conversion, no feature selection);
    text columns are one-hot encoded later in src/shared/core_utils.py
  - Price stays in MAD (no log transform), and no q2-q98 price range is applied after the split (unlike a2)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import full

RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_minimal.csv"
TARGET = "Price"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[full.preprocess(df).index].copy()  # same rows as the full preprocessing

    # Numeric features (only "Number of Doors" in the raw data): missing -> 0
    numeric = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df[numeric] = df[numeric].fillna(0)

    # Text features: missing -> "missing"
    text = [c for c in df.columns if c not in numeric + [TARGET]]
    df[text] = df[text].fillna("missing")
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
