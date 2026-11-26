"""European listings (Part B step 3): data/external/european_used_cars_100k.csv -> data/processed/cars_europe.csv

New, unseen data for the saved models (src/external/evaluate_saved_models.py), in the same columns as cars_full.csv.
Source: KTHESAT, "European Used Cars 100k Sample", Hugging Face, CC BY-NC 4.0 (mobile.de / AutoScout24, prices in euros).

Prices are converted to MAD and get the same price rules as full.py; brand / model names, fuel, gearbox, mileage and
equipment are mapped to the Moroccan form. Fiscal Power, Condition and Doors are not in the data and are filled by the
saved model. Only car models seen in training and prices inside the training price range are kept (training_reference).
See REPORT.md section 10.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import full

RAW_CSV = ROOT_DIR / "data" / "external" / "european_used_cars_100k.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "cars_europe.csv"
DOWNLOAD_URL = "https://huggingface.co/datasets/KTHESAT/european-used-cars-100k-sample"

# Euro -> dirham. One fixed rate for every row, so R2 and MAPE do not depend on it. 2026 average about 10.76-10.77 MAD per
# euro (wise.com / valutafx.com, range 10.59-10.94). Update the rate (and this comment) if you use another one.
MAD_PER_EUR = 10.77
ROUND_PRICE_TO = 10      # MAD
COLLECTION_YEAR = 2026   # snapshot of October 2026: later registration years are typing errors

MIN_SEEN = 10  # training cars a Brand + Model needs to count as seen (= OneHotEncoder min_frequency in core_utils)

BRAND_NAMES = {"MINI": "Mini", "SEAT": "Seat", "CUPRA": "Cupra", "Andere": "Autres", "Sonstige": "Autres"}

# Moroccan equipment item -> words that mark it in the (German) equipment list, lower case
EQUIPMENT = {
    "Electric Windows": ["fensterheber"],
    "CD/MP3/Bluetooth": ["bluetooth", "mp3", "cd"],
    "Alloy Wheels": ["leichtmetallfelgen"],
    "Air Conditioning": ["klima"],
    "Airbags": ["airbag"],
    "Central Locking": ["zentralverriegelung"],
    "Rear Camera": ["kamera"],
    "ABS": ["abs"],
    "Leather Seats": ["lederausstattung", "teilleder", "vollleder"],
    "Speed Limiter": ["geschwindigkeitsbegrenzer"],
    "Cruise Control": ["tempomat"],
    "Parking Sensors": ["einparkhilfe", "parksensor"],
    "Onboard Computer": ["bordcomputer"],
    "ESP": ["esp"],
    "Navigation System/GPS": ["navigationssystem"],
    "Sunroof": ["schiebedach", "panorama"],
}


def clean_model_name(brand, model):
    """Model name in the Moroccan form (full.clean_model first). Only the frequent cases where the two portals name the same
    car differently; an unknown model is still fine (the encoder treats it like a rare model)."""
    m = full.clean_model(brand, model)
    m = re.sub(r"\s*\(.*?\)", "", m).replace(" variant", "").strip()       # 'aygo (x)' -> 'aygo', 'passat variant' -> 'passat'
    if brand == "BMW" and re.fullmatch(r"[1-8]\d\d[a-z]*", m):           # '320', '118i' -> 'serie 3', 'serie 1'
        return f"serie {m[0]}"
    if brand == "Mercedes-Benz":
        found = re.fullmatch(r"(a|b|c|e|s|g|v|cla|cls|gla|glc|gle|gls|glk|ml|sl|slk)( klasse|\s?\d+.*)", m)
        if found:                                                        # 'c 220', 'c klasse' -> 'classe c'
            return f"classe {found.group(1)}"
    if brand == "Mini":
        if "countryman" in m:
            return "country man"
        if "cabrio" in m:
            return "cabrio"
        if "cooper" in m:                                                # 'cooper s', 'john cooper works' -> 'cooper'
            return "cooper"
    if brand == "MG" and m in ("hs", "ehs"):
        return "mg hs" if m == "hs" else "hs phev"
    return m


def main_fuel(text):
    """'Hybrid (Benzin/Elektro), E10-geeignet' -> 'Hybrid'; 'Benzin' / 'Super 95' -> 'Petrol'; 'Strom' -> 'Electric'.
    AutoScout24 sometimes stores a one-letter code (b = petrol, d = diesel, e = electric, l = LPG)."""
    t = str(text).lower()
    if t in ("b", "d", "e", "l"):
        return {"b": "Petrol", "d": "Diesel", "e": "Electric", "l": "LPG"}[t]
    if "hybrid" in t or ("/" in t and re.search(r"elektr|electr", t)):   # 'Elektro/Benzin', 'Electro/Gasolina'
        return "Hybrid"
    if re.search(r"elektr|electr|el.ctrico|strom", t):
        return "Electric"
    if re.search(r"lpg|glp|autogas|cng|erdgas|aardgas", t):
        return "LPG"
    if "diesel" in t:
        return "Diesel"
    if re.search(r"benzin|super|gasolina", t):
        return "Petrol"
    return np.nan  # 'Andere', 'unknown', hydrogen, ethanol ...


def gearbox(text):
    t = str(text).lower()
    if "auto" in t:  # Automatik, Halbautomatik, Boîte automatique, Half/Semi-automaat
        return 1
    if "schalt" in t or "manu" in t:
        return 0
    return np.nan


def equipment_amount(text):
    """Number of the Moroccan data's 16 equipment items found in the listing's list; NaN if none is found."""
    try:
        items = [str(i).lower() for i in json.loads(text)]
    except (TypeError, ValueError):
        return np.nan
    found = sum(any(word in item for item in items for word in words) for words in EQUIPMENT.values())
    return found if found else np.nan


def training_reference():
    """From the training rows of cars_full.csv (same split and price range as core_utils.load_data_full, i.e. the rows the
    saved a2 / b1 models were trained on): the Brand + Model pairs with at least MIN_SEEN cars, and the log-price range."""
    from src.shared.core_utils import PRICE_QUANTILES, read_processed, split_rows
    cars = read_processed(full.OUT_PATH, full.main)
    X_train, y_train, *_ = split_rows(cars.drop(columns="Log Price"), cars[["Log Price"]])
    low, high = y_train["Log Price"].quantile(list(PRICE_QUANTILES))
    X_train = X_train[y_train["Log Price"].between(low, high)]
    counts = X_train.groupby(["Brand", "Model"]).size()
    return set(counts[counts >= MIN_SEEN].index), (low, high)


def mileage_range(km) -> str:
    """89000 -> '85 000 - 89 999': km in this dataset's ranges (5k wide below 100k, 10k up to 200k, 50k up to 500k).
    Used for the external datasets, which give the mileage in km."""
    km = int(km)
    if km >= 500_000:
        return "Plus de 500 000"
    width = 5_000 if km < 100_000 else 10_000 if km < 200_000 else 50_000
    low = km // width * width
    return f"{low:,} - {low + width - 1:,}".replace(",", " ")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price: euros -> MAD (rounded to 10 MAD), same price rules as full.py (placeholder test on the original euro value)
    df = df.dropna(subset=["price_eur"])
    df["Price"] = (df["price_eur"] * MAD_PER_EUR / ROUND_PRICE_TO).round().astype(int) * ROUND_PRICE_TO
    df = df[df["Price"].between(full.MIN_PRICE, full.MAX_PRICE)].copy()
    df = df[~df["price_eur"].round().astype(int).map(full.is_placeholder_price)]

    # Brand / Model / Year
    df = df.dropna(subset=["make", "model"])
    df = df[df["first_registration_year"] <= COLLECTION_YEAR]
    df["Brand"] = df["make"].map(full.clean_brand).replace(BRAND_NAMES)
    df["Model"] = [clean_model_name(b, m) for b, m in zip(df["Brand"], df["model"])]
    df["Year"] = df["first_registration_year"].astype(int)

    # Price far from similar cars (same rule and tolerance as full.py)
    df = full.drop_price_outliers(df)

    # Columns of cars_full.csv
    df["Fuel"] = df["fuel_type"].map(main_fuel)
    df["Gearbox"] = df["transmission"].map(gearbox)
    df["Mileage Above 500000"] = (df["mileage_km"] >= 500_000).astype(int)
    df["Mileage"] = df["mileage_km"].map(mileage_range).map(full.mileage_midpoint)
    df["Equipment Amount"] = df["equipment"].map(equipment_amount)
    df["Fiscal Power Above 41"] = 0
    for missing in ("Condition", "Number of Doors", "Fiscal Power (CV)"):  # not in this dataset
        df[missing] = np.nan

    df = df.dropna(subset=["Fuel", "Gearbox"])
    seen, (low, high) = training_reference()
    df = df[[pair in seen for pair in zip(df["Brand"], df["Model"])]].copy()  # only car models the saved models were trained on
    df["Log Price"] = np.log1p(df["Price"])
    df = df[df["Log Price"].between(low, high)]  # only prices inside the training price range
    return df[full.FEATURE_COLUMNS + ["Log Price"]].drop_duplicates()


def main() -> None:
    if not RAW_CSV.exists():
        sys.exit(f"Download european_used_cars_100k.csv from {DOWNLOAD_URL} and save it as {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df_clean = preprocess(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    print(f"[Europe preprocessing] Rows: {len(df):,} -> {len(df_clean):,} (car models seen in training, prices in the training price range)")
    print(f"Saved {OUT_PATH}")
    print(f"Price (MAD, 1 EUR = {MAD_PER_EUR} MAD): median {np.expm1(df_clean['Log Price']).median():,.0f}")
    print("Missing values (filled by the saved model's preprocessing, learned from cars_full.csv):")
    print(df_clean.isna().sum()[lambda s: s > 0].to_string() or "  none")


if __name__ == "__main__":
    main()
