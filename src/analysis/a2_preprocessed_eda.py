"""
Exploratory Data Analysis script for data/raw/cars_dataframe.csv
Saves visual reports to reports/figures/
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # save to file only; no window pops up
ROOT_DIR = Path(__file__).resolve().parents[2]
FULL_CSV = ROOT_DIR / "data" / "processed" / "cars_full.csv"
FIG_DIR = ROOT_DIR / "reports" / "a2_preprocessed"


def plot_correlation_heatmap() -> None:
    """Pearson correlation of the numeric features with Log Price (fully preprocessed data, before imputation)."""
    print("=" * 60)
    print("CORRELATION HEATMAP (fully preprocessed data)")
    print("=" * 60)
    if not FULL_CSV.exists():
        from src.preprocessing import full
        full.main()
    df = pd.read_csv(FULL_CSV)

    numeric_cols = ["Year", "Mileage", "Condition", "Gearbox", "Number of Doors", "Equipment Amount", "Fiscal Power (CV)"]
    corr_matrix = df[numeric_cols + ["Log Price"]].corr(method="pearson")

    plt.figure(figsize=(10, 8))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"label": "Pearson Correlation (r)"})
    plt.title("Correlation Matrix of Numerical Features vs. Log Price", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    heatmap_path = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved Heatmap: {heatmap_path}\n")


def main():
    plot_correlation_heatmap()


if __name__ == "__main__":
    main()