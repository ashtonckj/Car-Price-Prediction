import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# File paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "cars.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_correlation_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    # 1. Select numeric/ordinal columns and add Log Price
    numeric_cols = [
        "Year", "Mileage", "Condition", "Gearbox", 
        "Number of Doors", "Equipment Amount", "Fiscal Power (CV)"
    ]
    
    # Create temp copy with Log Price
    heatmap_df = df[numeric_cols].copy()
    heatmap_df["Log Price"] = df["Log Price"]
    
    # 2. Compute Pearson Correlation Matrix
    corr_matrix = heatmap_df.corr(method="pearson")
    
    # 3. Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation (r)"}
    )
    plt.title("Correlation Matrix of Numerical Features vs. Log Price", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    
    heatmap_path = fig_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved Heatmap: {heatmap_path}")

def main():
    print("Starting EDA for Processed Dataset...\n")
    """Loads the processed dataframe without modifying it."""
    df = pd.read_csv(PROCESSED_CSV)

    plot_correlation_heatmap(df, FIG_DIR)


if __name__ == "__main__":
    main()