"""Minimal preprocessing (Part A, a1_baseline): data/raw/cars_dataframe.csv -> data/processed/cars_minimal.csv

Same rows as full.py (so a1 and a2 are scored on the same cars), but the columns are left almost raw:
missing numbers -> 0, missing text -> "missing", text one-hot encoded later in core_utils, price kept in MAD.
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
