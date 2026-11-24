"""Full preprocessing (Part A, step 5): data/raw/cars_dataframe.csv -> data/processed/cars_full.csv"""
import ast
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_full.csv"

# Price limits (MAD): anything outside is treated as a typo / placeholder
MIN_PRICE = 5_000
MAX_PRICE = 2_000_000
PRICE_TOLERANCE = 1.5
MIN_GROUP_SIZE = 5
BRAND_CODES = {
    "111": "DFSK", "97": "MG", "75": "Great Wall", "77": "Tata", "73": "Abarth",
    "72": "Lada", "76": "Lifan", "112": "Leapmotor", "74": "Autres", "99": "Autres",
}


def clean_brand(brand: str) -> str:
    # '97' -> 'MG' ; 'mini' -> 'Mini' ; 'lancia' -> 'Lancia' ; 'BMW' stays 'BMW'
    brand = str(brand).strip()
    if brand in BRAND_CODES:
        return BRAND_CODES[brand]
    return brand if brand[:1].isupper() else brand.title()


def clean_model(brand: str, model: str) -> str:
    # lowercase, remove accents, turn '_' and '-' into spaces: 'serie_3_gt' / 'Serie 3 GT' -> 'serie 3 gt'
    text = unicodedata.normalize("NFKD", str(model)).encode("ascii", "ignore").decode().lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Renault: 'R19' and '19' are the same car -> '19' (same for R4, R5, R9, R11 ...)
    if brand == "Renault":
        text = re.sub(r"^r ?(\d+)$", r"\1", text)
    return text


def mileage_midpoint(text: str) -> float:
    # '200 000 - 249 999' -> 225000 (middle of the range). 'Plus de 500 000' has no upper limit, so it becomes 500000
    if text.startswith("Plus de"):
        return 500_000
    low, high = text.replace(" ", "").split("-")
    return round((int(low) + int(high)) / 2)


def drop_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Compare every car with similar cars: remove it if its price is far from the typical price of its
    # Brand + Model, after taking out the general "older = cheaper" trend. Catches typos such as a
    # 2021 Dacia Sandero at 1,550,000 or a 2011 Touareg at 3,000 that a fixed MIN/MAX cannot see.
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


def is_placeholder_price(price: int) -> bool:
    # 4+ copies of the same digit, e.g. 8888, 11111, 999999 (but not 9999 / 99999, which are real prices)
    text = str(price)
    return len(text) >= 4 and len(set(text)) == 1 and price not in (9_999, 99_999)


FEATURE_COLUMNS = ["Brand", "Model", "Fuel", "Year", "Condition", "Gearbox", "Number of Doors", "Equipment Amount",
                   "Mileage", "Mileage Above 500000", "Fiscal Power (CV)", "Fiscal Power Above 41"]
CONDITION_ORDER = {"For Parts": 0, "Damaged": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5, "New": 6}


def clean_names_and_year(df: pd.DataFrame) -> pd.DataFrame:
    """Brand / Model text cleaning and Year as a number. Used for the training rows and for new (real-world) listings."""
    df = df.copy()
    df["Brand"] = df["Brand"].map(clean_brand)
    df["Model"] = [clean_model(b, m) for b, m in zip(df["Brand"], df["Model"])]
    # Year: "1980 ou plus ancien" -> 1980, and make it a number
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    return df


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> the model's feature columns (FEATURE_COLUMNS). Row-by-row only, nothing learned from the data,
    so the same function turns a new listing into features. Rows are not removed here (Condition may come out NaN)."""
    df = df.copy()
    # Mileage: range text -> number (middle of the range) plus a flag column for the open-ended "Plus de 500 000" bucket
    df["Mileage Above 500000"] = (df["Mileage"] == "Plus de 500 000").astype(int)
    df["Mileage"] = df["Mileage"].map(mileage_midpoint)

    # Condition: ordinal encoding (unknown / missing -> NaN)
    df["Condition"] = df["Condition"].map(CONDITION_ORDER)

    # Gearbox: Manual = 0, Automatic = 1
    df["Gearbox"] = df["Gearbox"].map({"Manual": 0, "Automatic": 1})

    # Fiscal Power: "8 CV" -> 8 ; "Plus de 41 CV" -> 41 plus a flag column (missing values stay NaN)
    df["Fiscal Power Above 41"] = (df["Fiscal Power"] == "Plus de 41 CV").astype(int)
    df["Fiscal Power (CV)"] = df["Fiscal Power"].str.extract(r"(\d+)")[0].astype(float)

    # Fuel: keep as text
    df["Fuel"] = df["Fuel"].replace({"Electrique": "Electric"})

    # Equipment -> Equipment Amount (number of items in the list)
    df["Equipment Amount"] = df["Equipment"].map(lambda x: len(ast.literal_eval(x)))

    # Number of Doors: missing values stay NaN. Every other raw column (First Owner, Origin, Location, Sector) is dropped.
    return df[FEATURE_COLUMNS]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the one corrupted row (a mileage range sitting in the Fuel column)
    df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])].copy()

    # Price: remove rows without a price (we never invent a target value)
    df = df.dropna(subset=["Price"]).copy()

    # Brand / Model: no missing values allowed, then clean the text (numeric brand codes, case, R19 -> 19 ...)
    df = df.dropna(subset=["Brand", "Model"]).copy()

    # Price: remove impossible values (e.g. 1, 24, 180 or 1,600,000,000)
    df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)].copy()

    # Price: whole numbers only (it was a float because of the missing values)
    df["Price"] = df["Price"].astype(int)

    # Price: remove placeholder prices made of one repeated digit (11111, 222222, 999999, 1111111 ...).
    # 9999 and 99999 are kept: they sit near the usual price of their model ("just under 10k / 100k" pricing)
    df = df[~df["Price"].map(is_placeholder_price)].copy()

    # Brand / Model text and Year (shared with new listings)
    df = clean_names_and_year(df)

    # Price: remove prices that are far away from similar cars
    df = drop_price_outliers(df)

    # Raw columns -> feature columns (shared with new listings); rows with a missing Condition are removed
    price = df["Price"]
    df = encode_columns(df)
    df = df.dropna(subset=["Condition"])

    # Price -> log1p (the target column; use np.expm1 to get back to MAD)
    df["Log Price"] = np.log1p(price.loc[df.index])

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
