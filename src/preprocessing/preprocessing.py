"""
preprocessing.py
Reads data/raw/cars_dataframe.csv and writes data/processed/cars.csv
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"

# Price limits (MAD): anything outside is treated as a typo / placeholder
MIN_PRICE = 3_000
MAX_PRICE = 3_000_000


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the one corrupted row (a mileage range sitting in the Fuel column)
    df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])]

    # Price: remove rows without a price
    df = df.dropna(subset=["Price"]).copy()

    # Price: remove impossible values (e.g. 1, 24, 180 or 1,600,000,000)
    df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)].copy()

    # Year: "1980 ou plus ancien" -> 1980, and make it a number
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

    # Condition: ordinal encoding (missing values stay empty)
    condition_order = {"For Parts": 0, "Damaged": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5, "New": 6}
    df["Condition"] = df["Condition"].map(condition_order)

    # Gearbox: Manual = 0, Automatic = 1
    df["Gearbox"] = df["Gearbox"].map({"Manual": 0, "Automatic": 1})

    # Fiscal Power: "Plus de 41 CV" -> ">41 CV"
    df["Fiscal Power"] = df["Fiscal Power"].replace({"Plus de 41 CV": ">41 CV"})

    # Fuel: one-hot encoding (only 5 categories)
    df["Fuel"] = df["Fuel"].replace({"Electrique": "Electric"})
    df = pd.get_dummies(df, columns=["Fuel"], dtype=int)

    # Equipment -> Equipment Amount (number of items in the list)
    df["Equipment Amount"] = df["Equipment"].map(lambda x: len(ast.literal_eval(x)))
    df = df.drop(columns=["Equipment"])

    # Number of Doors: fill with the most common value for that Brand + Model, else 5
    mode_by_model = df.groupby(["Brand", "Model"])["Number of Doors"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    df["Number of Doors"] = (df["Number of Doors"].fillna(mode_by_model).fillna(5).astype(int))

    # Drop columns that are not needed
    df = df.drop(columns=["Origin", "Location", "Sector"])

    return df


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    clean = preprocess(raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_DIR / "cars.csv", index=False)

    print(f"Rows: {len(raw)} -> {len(clean)}")
    print(f"Saved {OUT_DIR / "cars.csv"}")
    print("Remaining missing values:")
    print(clean.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()