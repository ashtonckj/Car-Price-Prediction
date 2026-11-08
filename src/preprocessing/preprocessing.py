"""
preprocessing.py
Reads data/raw/cars_dataframe.csv and writes data/processed/cars.csv
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"
OUT_PATH = OUT_DIR / "cars.csv"

# Price limits (MAD): anything outside is treated as a typo / placeholder
MIN_PRICE = 3_000
MAX_PRICE = 10_000_000

MIN_FREQUENCY = 10   # Brand/Model values with fewer rows are grouped into one "infrequent" column
CATEGORICAL_COLUMNS = ["Brand", "Model"]

def mileage_midpoint(text: str) -> float:
    # '200 000 - 249 999' -> 225000 (middle of the range). 'Plus de 500 000' has no upper limit, so it becomes 500000
    if text.startswith("Plus de"):
        return 500_000
    low, high = text.replace(" ", "").split("-")
    return round((int(low) + int(high)) / 2)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the one corrupted row (a mileage range sitting in the Fuel column)
    df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])]

    # Price: remove rows without a price
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

    # Condition: ordinal encoding (missing values are removed)
    CONDITION_ORDER = {"For Parts": 0, "Damaged": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5, "New": 6}
    df["Condition"] = df["Condition"].map(CONDITION_ORDER)
    df = df.dropna(subset=["Condition"])

    # Gearbox: Manual = 0, Automatic = 1
    df["Gearbox"] = df["Gearbox"].map({"Manual": 0, "Automatic": 1})

    # Fiscal Power: "8 CV" -> 8 ; "Plus de 41 CV" -> 41 plus a flag column (missing values stay empty and get flag 0)
    df["Fiscal Power Above 41"] = (df["Fiscal Power"] == "Plus de 41 CV").astype(int)
    df["Fiscal Power (CV)"] = df["Fiscal Power"].str.extract(r"(\d+)")[0].astype("Int64")
    df = df.drop(columns=["Fiscal Power"])

    # Fiscal Power: fill missing with the most common value of the same
    # Brand + Model + Fuel, then (for what is left) the same Brand + Model
    for keys in (["Brand", "Model", "Fuel"], ["Brand", "Model"]):
        group_mode = df.groupby(keys)["Fiscal Power (CV)"].transform(
            lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        df["Fiscal Power (CV)"] = df["Fiscal Power (CV)"].fillna(group_mode)


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
    df = df.drop(columns=["First Owner", "Origin", "Location", "Sector"])

    # Column order: general columns, Mileage + flag, Fiscal Power + flag, Fuel, Price
    fuel_cols = [c for c in df.columns if c.startswith("Fuel_")]
    df = df[["Brand", "Model", "Year", "Condition", "Gearbox", "Number of Doors", "Equipment Amount",
             "Mileage", "Mileage Above 500000", "Fiscal Power (CV)", "Fiscal Power Above 41", *fuel_cols, "Price"]]

    df = df.drop_duplicates()

    # Brand + Model: one-hot encoding; values with fewer than MIN_FREQUENCY rows share one "infrequent" column
    encoder = OneHotEncoder(min_frequency=MIN_FREQUENCY, handle_unknown="infrequent_if_exist", sparse_output=False, dtype=int)
    encoded = pd.DataFrame(
        encoder.fit_transform(df[CATEGORICAL_COLUMNS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=CATEGORICAL_COLUMNS), encoded], axis=1)

    # Price -> log1p (the target column; use np.expm1 to get back to MAD)
    df["Log Price"] = np.log1p(df.pop("Price"))

    return df


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    clean = preprocess(raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_PATH, index=False)

    print(f"Rows: {len(raw)} -> {len(clean)}")
    print(f"Saved {OUT_PATH}")
    print("Remaining missing values:")
    print(clean.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()