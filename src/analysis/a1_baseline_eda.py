"""
Exploratory Data Analysis script for data/raw/cars_dataframe.csv
Saves visual reports to reports/a1_baseline/
"""
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT_DIR / "data" / "raw" / "cars_dataframe.csv"
FIG_DIR = ROOT_DIR / "reports" / "a1_baseline"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_raw_data() -> pd.DataFrame:
    """Loads the raw dataframe without modifying it."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_CSV}")
    return pd.read_csv(RAW_CSV)

def prepare_numerical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and parses all continuous features into numeric dtypes for outlier testing."""
    num_df = pd.DataFrame()

    # 1. Price
    if "Price" in df.columns:
        num_df["Price"] = pd.to_numeric(
            df["Price"].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce"
        )

    # 2. Year
    if "Year" in df.columns:
        num_df["Year"] = pd.to_numeric(
            df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
        )

    # 3. Fiscal Power
    if "Fiscal Power" in df.columns:
        num_df["Fiscal Power"] = pd.to_numeric(
            df["Fiscal Power"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )

    # 4. Equipment Amount
    if "Equipment" in df.columns:
        num_df["Equipment Amount"] = df["Equipment"].apply(
            lambda x: len(ast.literal_eval(x)) if pd.notna(x) and isinstance(x, str) and x.startswith("[") else 0
        )

    # 5. Number of Doors
    if "Number of Doors" in df.columns:
        num_df["Number of Doors"] = pd.to_numeric(df["Number of Doors"], errors="coerce")

    return num_df


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
        print("-" * 40)

    # Brand anomalies (numeric IDs, missing values)
    if "Brand" in df.columns:
        print(f"Brand Summary ({df['Brand'].nunique():,} unique brands):")
        print("Top 10 Brands:")
        print(df["Brand"].value_counts(dropna=False).head(10).to_string())
        brand_str = df["Brand"].dropna().astype(str)
        numeric_brands = brand_str[brand_str.str.isnumeric()]
        print(f"Corrupted / Numeric Brand IDs: {len(numeric_brands):,} rows (e.g., {numeric_brands.unique()[:5].tolist()})")
        print(f"Missing Brand: {df['Brand'].isna().sum():,} rows")
        print("-" * 40)

    # Model anomalies (slug format, numeric values, missing values)
    if "Model" in df.columns:
        print(f"Model Summary ({df['Model'].nunique():,} unique models):")
        print("Top 10 Models:")
        print(df["Model"].value_counts(dropna=False).head(10).to_string())
        model_str = df["Model"].dropna().astype(str)
        slug_models = model_str[model_str.str.contains("_", na=False)]
        print(f"Slug / Underscore Formatted Models: {len(slug_models):,} rows (e.g., {slug_models.unique()[:5].tolist()})")
        print(f"Missing Model: {df['Model'].isna().sum():,} rows")
        print("-" * 40)

    # Origin categories and missingness
    if "Origin" in df.columns:
        print("Origin Value Counts:")
        print(df["Origin"].value_counts(dropna=False).to_string())
        missing_origin = df["Origin"].isna().sum()
        print(f"Missing Origin: {missing_origin:,} ({missing_origin / len(df) * 100:.2f}%)")
        print("-" * 40)

    # First Owner categories and missingness
    if "First Owner" in df.columns:
        print("First Owner Value Counts:")
        print(df["First Owner"].value_counts(dropna=False).to_string())
        missing_fo = df["First Owner"].isna().sum()
        print(f"Missing First Owner: {missing_fo:,} ({missing_fo / len(df) * 100:.2f}%)")
        print("-" * 40)

    # Location (cardinality, rare categories, Arabic vs Latin script mismatch)
    if "Location" in df.columns:
        print(f"Location Summary ({df['Location'].nunique():,} unique locations):")
        print("Top 10 Locations:")
        print(df["Location"].value_counts(dropna=False).head(10).to_string())
        loc_series = df["Location"].dropna().astype(str)
        arabic_locs = loc_series[loc_series.apply(lambda s: any('\u0600' <= c <= '\u06ff' for c in s))]
        print(f"Non-Latin / Arabic Script Locations: {len(arabic_locs):,} rows ({arabic_locs.unique().tolist()})")
        rare_locs = (df["Location"].value_counts() == 1).sum()
        print(f"Rare Locations (Only 1 listing): {rare_locs:,}")
        print("-" * 40)

    # Sector (high cardinality, placeholders, missingness)
    if "Sector" in df.columns:
        print(f"Sector Summary ({df['Sector'].nunique():,} unique sectors, Missing: {df['Sector'].isna().sum():,}):")
        print("Top 10 Sectors:")
        print(df["Sector"].value_counts(dropna=False).head(10).to_string())
        generic_sectors = ["Toute la ville", "City Center", "Other", "Autre"]
        generic_count = df["Sector"].isin(generic_sectors).sum()
        print(f"Generic / Placeholder Sectors: {generic_count:,} ({generic_count / len(df) * 100:.2f}%)")
        print("-" * 40)

    # Mileage (range format, open-ended bucket, anomalies, missingness)
    if "Mileage" in df.columns:
        print(f"Mileage Summary ({df['Mileage'].nunique():,} unique range formats):")
        print("Top 10 Mileage Buckets:")
        print(df["Mileage"].value_counts(dropna=False).head(10).to_string())
        mileage_series = df["Mileage"].dropna().astype(str)
        has_range = mileage_series.str.contains("-", na=False)
        has_plus = mileage_series.str.startswith("Plus de")
        non_standard = mileage_series[~has_range & ~has_plus]
        print(f"Standard Ranges ('X - Y'): {has_range.sum():,}")
        print(f"Open-Ended Bucket ('Plus de 500 000'): {has_plus.sum():,}")
        if len(non_standard) > 0:
            print(f"Non-standard / Anomalous Mileage Entries: {len(non_standard):,} (Values: {non_standard.tolist()})")
        print(f"Missing Mileage: {df['Mileage'].isna().sum():,} rows")
        print("-" * 40)

    # Number of Doors (discrete category distribution, missingness, unexpected values)
    if "Number of Doors" in df.columns:
        print("Number of Doors Value Counts:")
        print(df["Number of Doors"].value_counts(dropna=False).to_string())
        missing_doors = df["Number of Doors"].isna().sum()
        unexpected_doors = df["Number of Doors"].dropna()[~df["Number of Doors"].dropna().isin([3.0, 5.0, 3, 5])]
        print(f"Unexpected Door Values (outside 3 and 5): {len(unexpected_doors):,}")
        print(f"Missing Number of Doors: {missing_doors:,} ({missing_doors / len(df) * 100:.2f}%)")
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
        palette="viridis",
        hue="Brand",
        legend=False
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


def plot_all_boxplots(num_df: pd.DataFrame) -> None:
    """Generates boxplots for every numerical feature."""
    cols = num_df.columns
    n_cols = len(cols)

    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(n_cols / 2)), figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(cols):
        sns.boxplot(
            y=num_df[col],
            ax=axes[idx],
            color="skyblue",
            flierprops={"markerfacecolor": "red", "markersize": 4}
        )
        axes[idx].set_title(f"Boxplot: {col}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("")

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    boxplot_path = FIG_DIR / "all_features_boxplots.png"
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saved boxplots: {boxplot_path}")


def plot_z_scores(num_df: pd.DataFrame) -> None:
    """Plots Z-score distributions with threshold lines at Z = -3 and Z = +3."""
    cols = num_df.columns
    n_cols = len(cols)

    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(n_cols / 2)), figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(cols):
        s = num_df[col].dropna()
        if len(s) == 0 or s.std() == 0:
            continue

        z_scores = (s - s.mean()) / s.std()
        outliers_count = (z_scores.abs() > 3).sum()

        sns.histplot(z_scores, kde=True, ax=axes[idx], color="mediumpurple", bins=40)
        axes[idx].axvline(3, color="red", linestyle="--", linewidth=1.5, label="Z = +3")
        axes[idx].axvline(-3, color="red", linestyle="--", linewidth=1.5, label="Z = -3")
        
        axes[idx].set_title(f"Z-Score Distribution: {col}\n({outliers_count:,} outliers |Z| > 3)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Z-Score")
        axes[idx].set_ylabel("Count")
        axes[idx].legend(loc="upper right")

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    z_plot_path = FIG_DIR / "all_features_z_scores.png"
    plt.savefig(z_plot_path)
    plt.close()
    print(f"Saved Z-Score distribution plots: {z_plot_path}")

def detect_outliers_summary(num_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates IQR and Z-Score outlier counts for each numerical feature."""
    summary_list = []

    for col in num_df.columns:
        s = num_df[col].dropna()
        n_total = len(s)

        if n_total == 0:
            continue

        # IQR Method
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_outliers = s[(s < lower_iqr) | (s > upper_iqr)].count()

        # Z-Score Method (|Z| > 3)
        mean = s.mean()
        std = s.std()
        if std > 0:
            z_scores = (s - mean) / std
            z_outliers = (z_scores.abs() > 3).sum()
        else:
            z_outliers = 0

        summary_list.append({
            "Feature": col,
            "Valid Count": n_total,
            "Mean": round(mean, 2),
            "Std Dev": round(std, 2),
            "IQR Outliers": iqr_outliers,
            "IQR Outliers (%)": round((iqr_outliers / n_total) * 100, 2),
            "Z-Score Outliers (|Z|>3)": z_outliers,
            "Z-Score Outliers (%)": round((z_outliers / n_total) * 100, 2),
        })

    return pd.DataFrame(summary_list)


def main():
    print("Starting EDA for Raw Dataset...\n")
    df = load_raw_data()
    num_df = prepare_numerical_df(df)
    
    inspect_overview(df)
    inspect_target(df)
    inspect_categorical_anomalies(df)
    #inspect_mileage_and_year(df)

    print("=" * 60)
    #plot_all_boxplots(num_df)
    #plot_z_scores(num_df)
    outlier_table = detect_outliers_summary(num_df)
    print("\nStatistical Outlier Summary:")
    print(outlier_table.to_string(index=False))

    #inspect_bivariate_relationships(df)

    print("=" * 60)
    print(f"EDA Complete! Diagnostic charts saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()