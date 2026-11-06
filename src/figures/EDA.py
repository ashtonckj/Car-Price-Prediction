"""
eda.py
Exploratory Data Analysis script for data/raw/cars_dataframe.csv
Saves visual reports to reports/figures/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# File paths
ROOT_DIR = Path(__file__).resolve().parent
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_raw_data() -> pd.DataFrame:
    """Loads the raw dataframe without modifying it."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_CSV}")
    return pd.read_csv(RAW_CSV)


def inspect_overview(df: pd.DataFrame) -> None:
    """1. General Data Overview & Missingness Check"""
    print("=" * 60)
    print(" 1. DATASET OVERVIEW & MISSING VALUE SUMMARY")
    print("=" * 60)
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"Duplicate Rows: {df.duplicated().sum():,}\n")

    print("Column Data Types & Missingness:")
    missing_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().sum() / len(df) * 100).round(2)
    }).sort_values(by="Missing Count", ascending=False)
    print(missing_df.to_string())
    print("\n")


def inspect_target(df: pd.DataFrame) -> None:
    """2. Target Variable Analysis (Price)"""
    print("=" * 60)
    print(" 2. TARGET ANALYSIS (Price)")
    print("=" * 60)

    # Clean numeric conversion for analysis
    price_cleaned = pd.to_numeric(
        df["Price"].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce"
    )

    print(f"Missing Prices: {price_cleaned.isna().sum():,}")
    print("Price Quantiles (RAW):")
    print(price_cleaned.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]).apply(lambda x: f"{x:,.0f} MAD"))

    # Count potential placeholder / corrupted prices
    extreme_low = (price_cleaned < 5_000).sum()
    extreme_high = (price_cleaned > 2_000_000).sum()
    print(f"\nSuspiciously Low Prices (< 5,000 MAD):   {extreme_low:,} rows")
    print(f"Suspiciously High Prices (> 2M MAD):      {extreme_high:,} rows")

    # Plot Price Distributions
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale plot (filtered to < 1M for readability)
    sns.histplot(price_cleaned[price_cleaned < 1_000_000], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Price Distribution (< 1,000,000 MAD)")
    axes[0].set_xlabel("Price (MAD)")

    # Log scale plot
    sns.histplot(np.log1p(price_cleaned.dropna()), bins=50, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Log(1 + Price) Distribution")
    axes[1].set_xlabel("Log(Price)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_price_distribution.png")
    plt.close()
    print(f"\nSaved plot: {FIG_DIR / 'target_price_distribution.png'}\n")


def inspect_categorical_anomalies(df: pd.DataFrame) -> None:
    """3. Categorical Values & Language Mismatch Check"""
    print("=" * 60)
    print(" 3. CATEGORICAL VALUES & ANOMALIES")
    print("=" * 60)

    # Fuel anomalies
    if "Fuel" in df.columns:
        print("Fuel Value Counts:")
        print(df["Fuel"].value_counts(dropna=False).to_string())
        print("-" * 40)

    # Gearbox language variants
    if "Gearbox" in df.columns:
        print("Gearbox Value Counts (Check for French vs English):")
        print(df["Gearbox"].value_counts(dropna=False).to_string())
        print("-" * 40)

    # Condition language variants
    if "Condition" in df.columns:
        print("Condition Value Counts:")
        print(df["Condition"].value_counts(dropna=False).to_string())
        print("-" * 40)

    # Fiscal Power values
    if "Fiscal Power" in df.columns:
        print("Fiscal Power Value Counts (Top 10):")
        print(df["Fiscal Power"].value_counts(dropna=False).head(10).to_string())
        print("\n")


def inspect_mileage_and_year(df: pd.DataFrame) -> None:
    """4. Mileage and Year Inspection"""
    print("=" * 60)
    print(" 4. MILEAGE & YEAR FORMAT INSPECTION")
    print("=" * 60)

    if "Mileage" in df.columns:
        # Sample non-standard Mileage values (ranges, non-numeric strings)
        sample_mileage = df["Mileage"].dropna().astype(str)
        ranges = sample_mileage[sample_mileage.str.contains("-", na=False)]
        print(f"Total Mileage entries containing ranges (e.g., '10000 - 20000'): {len(ranges):,}")
        print("Sample Range Strings:")
        print(ranges.head(5).to_list())
        print("-" * 40)

    if "Year" in df.columns:
        print("Year Unique Formats (Sample):")
        print(df["Year"].dropna().astype(str).unique()[:10])
        print("\n")


def inspect_bivariate_relationships(df: pd.DataFrame) -> None:
    """5. Feature-Target Relationships"""
    print("=" * 60)
    print(" 5. BIVARIATE RELATIONSHIPS & TOP BRANDS")
    print("=" * 60)

    # Filter out extreme target outliers for correlation analysis
    price_clean = pd.to_numeric(df["Price"].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    year_clean = pd.to_numeric(df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    
    analysis_df = pd.DataFrame({
        "Price": price_clean,
        "Year": year_clean,
        "Brand": df["Brand"] if "Brand" in df.columns else None
    }).dropna(subset=["Price"])
    analysis_df = analysis_df[analysis_df["Price"].between(10_000, 1_500_000)]

    # Top 15 Brands by listing count
    top_brands = analysis_df["Brand"].value_counts().head(15).index

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=analysis_df[analysis_df["Brand"].isin(top_brands)],
        x="Brand",
        y="Price",
        palette="viridis"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Price Distribution Across Top 15 Brands (10k - 1.5M MAD)")
    plt.ylabel("Price (MAD)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "price_by_brand_boxplot.png")
    plt.close()
    print(f"Saved plot: {FIG_DIR / 'price_by_brand_boxplot.png'}")

    # Price vs Year plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=analysis_df[analysis_df["Year"] >= 1990], x="Year", y="Price", alpha=0.3, color="teal")
    plt.title("Price vs. Manufacturing Year (1990+)")
    plt.ylabel("Price (MAD)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "price_vs_year.png")
    plt.close()
    print(f"Saved plot: {FIG_DIR / 'price_vs_year.png'}\n")


def main():
    print("Starting EDA for Raw Dataset...\n")
    df = load_raw_data()
    
    inspect_overview(df)
    inspect_target(df)
    inspect_categorical_anomalies(df)
    inspect_mileage_and_year(df)
    inspect_bivariate_relationships(df)
    
    print("=" * 60)
    print(f"EDA Complete! Diagnostic charts saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()