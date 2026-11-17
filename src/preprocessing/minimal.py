"""
Minimal preprocessing (Part A, step 1): data/raw/cars_dataframe.csv -> data/processed/cars_minimal.csv

The baseline uses the SAME ROWS as the full preprocessing (the rows full.py keeps, in the same order), so both
stages are trained and tested on the same cars and the comparison only measures what the column preprocessing adds.
The row-selection logic is copied below (section "Row selection") so this file does not import full.py.
IMPORTANT: if you change the row rules in full.py, make the same change here, otherwise the two files no longer
keep the same rows.

For the columns it deliberately does as little as possible:
  - Numeric feature missing -> 0 (a constant: nothing is learned from the data, so it is not imputation)
  - Text feature missing    -> the text "missing" (its own one-hot category, so no row is lost)
  - Every raw column is kept and stays as it is (no car-name cleaning, no mileage conversion, no feature selection);
    text columns are one-hot encoded later in src/shared/core_utils.py
  - Price stays in MAD (no log transform)
The price scope (q1-q99 of the training rows) is applied after the split in core_utils.py, as for a2.
"""
import ast
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_minimal.csv"
TARGET = "Price"

# ----------------------------------------------------------------------------------------------------------------------
# Row selection (copied from full.py: it only decides WHICH rows are kept, the cleaned values are thrown away)
# ----------------------------------------------------------------------------------------------------------------------

# Price limits (MAD): anything outside is treated as a typo / placeholder
MIN_PRICE = 5_000
MAX_PRICE = 2_000_000
PRICE_TOLERANCE = 1.5
MIN_GROUP_SIZE = 5
BRAND_CODES = {
    "111": "DFSK", "97": "MG", "75": "Great Wall", "77": "Tata", "73": "Abarth",
    "72": "Lada", "76": "Lifan", "112": "Leapmotor", "74": "Autres", "99": "Autres",
}


def _clean_brand(brand: str) -> str:
    brand = str(brand).strip()
    if brand in BRAND_CODES:
        return BRAND_CODES[brand]
    return brand if brand[:1].isupper() else brand.title()


def _clean_model(brand: str, model: str) -> str:
    text = unicodedata.normalize("NFKD", str(model)).encode("ascii", "ignore").decode().lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    if brand == "Renault":
        text = re.sub(r"^r ?(\d+)$", r"\1", text)
    return text


def _mileage_midpoint(text: str) -> float:
    if text.startswith("Plus de"):
        return 500_000
    low, high = text.replace(" ", "").split("-")
    return round((int(low) + int(high)) / 2)


def _is_placeholder_price(price: int) -> bool:
    text = str(price)
    return len(text) >= 4 and len(set(text)) == 1 and price not in (9_999, 99_999)


def _drop_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    log_price = np.log1p(df["Price"])
    slope = np.polyfit(df["Year"], log_price, 1)[0]
    adjusted = log_price - slope * df["Year"]

    key = df["Brand"] + "|" + df["Model"]
    model_median = adjusted.groupby(key).transform("median")
    model_count = adjusted.groupby(key).transform("count")
    brand_median = adjusted.groupby(df["Brand"]).transform("median")
    brand_count = adjusted.groupby(df["Brand"]).transform("count")

    typical = np.where(model_count >= MIN_GROUP_SIZE, model_median, np.where(brand_count >= MIN_GROUP_SIZE, brand_median, adjusted.median()))
    return df[(adjusted - typical).abs() <= PRICE_TOLERANCE]


def _kept_index(raw: pd.DataFrame) -> pd.Index:
    """Index of the raw rows that full.py keeps. Same steps, same order as full.preprocess."""
    df = raw.copy()

    df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])].copy()
    df = df.dropna(subset=["Price"]).copy()

    df = df.dropna(subset=["Brand", "Model"]).copy()
    df["Brand"] = df["Brand"].map(_clean_brand)
    df["Model"] = [_clean_model(b, m) for b, m in zip(df["Brand"], df["Model"])]

    df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)].copy()
    df["Price"] = df["Price"].astype(int)
    df = df[~df["Price"].map(_is_placeholder_price)].copy()

    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    df = _drop_price_outliers(df)

    df["Mileage Above 500000"] = (df["Mileage"] == "Plus de 500 000").astype(int)
    df["Mileage"] = df["Mileage"].map(_mileage_midpoint)

    CONDITION_ORDER = {"For Parts": 0, "Damaged": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5, "New": 6}
    df["Condition"] = df["Condition"].map(CONDITION_ORDER)
    df = df.dropna(subset=["Condition"])

    df["Gearbox"] = df["Gearbox"].map({"Manual": 0, "Automatic": 1})

    df["Fiscal Power Above 41"] = (df["Fiscal Power"] == "Plus de 41 CV").astype(int)
    df["Fiscal Power (CV)"] = df["Fiscal Power"].str.extract(r"(\d+)")[0].astype(float)

    df["Fuel"] = df["Fuel"].replace({"Electrique": "Electric"})
    df["Equipment Amount"] = df["Equipment"].map(lambda x: len(ast.literal_eval(x)))
    df["Log Price"] = np.log1p(df.pop("Price"))

    # full.py removes exact duplicates over exactly these columns, so we need them to find the same duplicates
    df = df[["Brand", "Model", "Fuel", "Year", "Condition", "Gearbox", "Number of Doors", "Equipment Amount",
             "Mileage", "Mileage Above 500000", "Fiscal Power (CV)", "Fiscal Power Above 41", "Log Price"]]
    return df.drop_duplicates().index


# ----------------------------------------------------------------------------------------------------------------------
# Minimal preprocessing
# ----------------------------------------------------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Same rows as the full preprocessing (its row rules: price 5k-2M, placeholder prices, duplicates, ...)
    df = df.loc[_kept_index(df)].copy()

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