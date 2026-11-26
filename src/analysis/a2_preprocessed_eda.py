"""Correlation heatmap of data/processed/cars_full.csv. Saves it to reports/a2_preprocessed/."""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")  # save to file only; no window pops up
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import full
from src.shared.core_utils import read_processed

FIG_DIR = ROOT_DIR / "reports" / "a2_preprocessed"


def plot_correlation_heatmap() -> None:
    """Pearson correlation of the numeric features with Log Price (fully preprocessed data, before imputation)."""
    print("=" * 60)
    print("CORRELATION HEATMAP (fully preprocessed data)")
    print("=" * 60)
    df = read_processed(full.OUT_PATH, full.main)

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
