"""Full preprocessing (Part A, step 5): data/raw/cars_dataframe.csv -> data/processed/cars_full.csv"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_full.csv"

# Price limits (MAD): anything outside is treated as a typo / placeholder
MIN_PRICE = 3_000
MAX_PRICE = 10_000_000


def mileage_midpoint(text: str) -> float:
    # '200 000 - 249 999' -> 225000 (middle of the range). 'Plus de 500 000' has no upper limit, so it becomes 500000
    if text.startswith("Plus de"):
        return 500_000
    low, high = text.replace(" ", "").split("-")
    return round((int(low) + int(high)) / 2)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the one corrupted row (a mileage range sitting in the Fuel column)
    df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])].copy()

    # Price: remove rows without a price (we never invent a target value)
    df = df.dropna(subset=["Price"]).copy()

    # Price: remove impossible values (e.g. 1, 24, 180 or 1,600,000,000)
    df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)].copy()

    # Price: whole numbers only (it was a float because of the missing values)
    df["Price"] = df["Price"].astype(int)

    # Year: "1980 ou plus ancien" -> 1980, and make it a number
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

    # Mileage: range text -> number (middle of the range) plus a flag column for the open-ended "Plus de 500 000" bucket
    df["Mileage Above 500000"] = (df["Mileage"] == "Plus de 500 000").astype(int)
    df["Mileage"] = df["Mileage"].map(mileage_midpoint)

    # Condition: ordinal encoding (rows with missing values are removed)
    CONDITION_ORDER = {"For Parts": 0, "Damaged": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5, "New": 6}
    df["Condition"] = df["Condition"].map(CONDITION_ORDER)
    df = df.dropna(subset=["Condition"])

    # Gearbox: Manual = 0, Automatic = 1
    df["Gearbox"] = df["Gearbox"].map({"Manual": 0, "Automatic": 1})

    # Fiscal Power: "8 CV" -> 8 ; "Plus de 41 CV" -> 41 plus a flag column (missing values stay NaN)
    df["Fiscal Power Above 41"] = (df["Fiscal Power"] == "Plus de 41 CV").astype(int)
    df["Fiscal Power (CV)"] = df["Fiscal Power"].str.extract(r"(\d+)")[0].astype(float)
    df = df.drop(columns=["Fiscal Power"])

    # Fuel: keep as text
    df["Fuel"] = df["Fuel"].replace({"Electrique": "Electric"})

    # Equipment -> Equipment Amount (number of items in the list)
    df["Equipment Amount"] = df["Equipment"].map(lambda x: len(ast.literal_eval(x)))
    df = df.drop(columns=["Equipment"])

    # Number of Doors: missing values stay NaN

    # Drop columns that are not needed
    df = df.drop(columns=["First Owner", "Origin", "Location", "Sector"])

    # Price -> log1p (the target column; use np.expm1 to get back to MAD)
    df["Log Price"] = np.log1p(df.pop("Price"))

    # Column order: categorical text columns, numeric columns, target
    df = df[["Brand", "Model", "Fuel", "Year", "Condition", "Gearbox", "Number of Doors", "Equipment Amount",
             "Mileage", "Mileage Above 500000", "Fiscal Power (CV)", "Fiscal Power Above 41", "Log Price"]]

    df = df.drop_duplicates()
    return df


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df_clean = preprocess(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    print(f"[Full preprocessing] Rows: {len(df):,} -> {len(df_clean):,}")
    print(f"Saved {OUT_PATH}")
    print("Remaining missing values (imputed later from the training rows only):")
    print(df_clean.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
