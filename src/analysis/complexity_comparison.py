"""Combines and compares complexity curve results between a2_preprocessed (default) and b1_tuned (tuned).

Outputs:
  - results/comparison/complexity_comparison.csv (side-by-side comparison table with differences & improvements)
  - results/comparison/complexity_comparison.png (4-panel comparison plot with Random Forest inset zoom)
"""
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.shared.core_utils import RESULTS_ROOT
from src.shared.plots import THOUSANDS
from src.shared import complexity as cc

OUT_DIR = RESULTS_ROOT / "comparison"

CURVE_METADATA = {
    "polynomial_degree": {
        "family": "Linear",
        "model": "Polynomial Regression",
        "parameter": "degree",
        "xlabel": "Polynomial degree (simple -> complex)",
        "categorical": True,
        "order": ["1", "2", "3", "4"],
        "default_tuned": None,
        "tuned_value": 2,
    },
    "random_forest_max_depth": {
        "family": "Tree",
        "model": "Random Forest",
        "parameter": "max_depth",
        "xlabel": "Maximum tree depth (simple -> complex)",
        "categorical": True,
        "order": ["10", "20", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "None"],
        "zoom_values": ["30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "None"],
        "default_tuned": None,
        "tuned_value": 50,
    },
    "knn_k": {
        "family": "Distance",
        "model": "KNN",
        "parameter": "n_neighbors",
        "xlabel": "Number of neighbours k (simple -> complex)",
        "categorical": True,
        "order": ["100", "75", "50", "30", "20", "15", "10", "9", "7", "5", "3", "1"],
        "default_tuned": None,
        "tuned_value": 9,
    },
    "dcnv2_epochs": {
        "family": "Neural",
        "model": "DCN-V2",
        "parameter": "epoch",
        "xlabel": "Training epochs (simple -> complex)",
        "categorical": False,
        "order": None,
        "default_tuned": None,
        "tuned_value": None,
    },
}


def load_stage_complexity(stage):
    path = RESULTS_ROOT / stage / "complexity_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
    df["value"] = df["value"].map(cc._clean_value)
    df["stage"] = stage
    return df


def plot_comparison_panels(df_a2, df_b1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, (curve_key, meta) in zip(axes.ravel(), CURVE_METADATA.items()):
        a2_sub = df_a2[df_a2["curve"] == curve_key].reset_index(drop=True)
        b1_sub = df_b1[df_b1["curve"] == curve_key].reset_index(drop=True)

        if a2_sub.empty and b1_sub.empty:
            ax.set_title(f"{meta['family']}: {meta['model']}")
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if meta["categorical"]:
            # Gather unique x values preserving defined order
            order = meta.get("order")
            all_vals = list(dict.fromkeys(list(a2_sub["value"]) + list(b1_sub["value"])))
            if order:
                x_vals = [v for v in order if v in all_vals] + [v for v in all_vals if v not in order]
            else:
                x_vals = all_vals

            # Re-index data to x_vals
            a2_map = {row["value"]: row for _, row in a2_sub.iterrows()}
            b1_map = {row["value"]: row for _, row in b1_sub.iterrows()}

            a2_train = [a2_map[v]["train_RMSE"] if v in a2_map else np.nan for v in x_vals]
            a2_val = [a2_map[v]["val_RMSE"] if v in a2_map else np.nan for v in x_vals]
            b1_train = [b1_map[v]["train_RMSE"] if v in b1_map else np.nan for v in x_vals]
            b1_val = [b1_map[v]["val_RMSE"] if v in b1_map else np.nan for v in x_vals]

            # Plot default (a2)
            ax.plot(x_vals, a2_train, "o--", color="#6baed6", linewidth=1.5, markersize=4, label="Default Train (a2)")
            ax.plot(x_vals, a2_val, "o--", color="#fd8d3c", linewidth=2.0, markersize=5, label="Default Val (a2)")

            # Plot tuned (b1)
            ax.plot(x_vals, b1_train, "s-", color="#08519c", linewidth=2.0, markersize=5, label="Tuned Train (b1)")
            ax.plot(x_vals, b1_val, "s-", color="#d94701", linewidth=2.5, markersize=6, label="Tuned Val (b1)")

            # Best validation point markings
            if not a2_sub.empty:
                best_a2_idx = int(a2_sub["val_RMSE"].idxmin())
                best_a2_val = a2_sub["value"][best_a2_idx]
                best_a2_score = a2_sub["val_RMSE"][best_a2_idx]
                ax.axvline(best_a2_val, color="#ff3c23", linestyle=":", alpha=0.8,
                           label=f"Best Default: {best_a2_val} ({best_a2_score:,.0f} MAD)")

            if not b1_sub.empty:
                best_b1_idx = int(b1_sub["val_RMSE"].idxmin())
                best_b1_val = b1_sub["value"][best_b1_idx]
                best_b1_score = b1_sub["val_RMSE"][best_b1_idx]
                ax.axvline(best_b1_val, color="#2ca02c", linestyle="-.", alpha=0.9,
                           label=f"Best Tuned: {best_b1_val} ({best_b1_score:,.0f} MAD)")

            # Inset zoom for Random Forest to inspect the plateau validation comparison
            if curve_key == "random_forest_max_depth" and not a2_sub.empty and not b1_sub.empty:
                try:
                    zoom_targets = meta.get("zoom_values", ["40", "50", "60", "70", "80", "90", "100", "None"])
                    zx = [v for v in x_vals if v in zoom_targets]
                    za2_val = [a2_map[v]["val_RMSE"] if v in a2_map else np.nan for v in zx]
                    zb1_val = [b1_map[v]["val_RMSE"] if v in b1_map else np.nan for v in zx]

                    ax_ins = ax.inset_axes([0.67, 0.68, 0.32, 0.3])
                    ax_ins.plot(zx, za2_val, "o--", color="#fd8d3c", linewidth=1.8, markersize=4, label="Default Val")
                    ax_ins.plot(zx, zb1_val, "s-", color="#d94701", linewidth=2.0, markersize=5, label="Tuned Val")

                    # Highlight best tuned in zoom
                    z_best_b1 = int(b1_sub["val_RMSE"].idxmin())
                    if b1_sub["value"][z_best_b1] in zx:
                        ax_ins.axvline(b1_sub["value"][z_best_b1], color="#2ca02c", linestyle=":", alpha=0.8)

                    Z_best_a2 = int(a2_sub["val_RMSE"].idxmin())
                    if a2_sub["value"][Z_best_a2] in zx:
                        ax_ins.axvline(a2_sub["value"][Z_best_a2], color="#fe3731", linestyle=":", alpha=0.8)

                    valid_vals = [v for v in za2_val + zb1_val if not np.isnan(v)]
                    if valid_vals:
                        v_min, v_max = min(valid_vals), max(valid_vals)
                        v_span = v_max - v_min
                        ax_ins.set_ylim(v_min - max(v_span * 0.25, 30), v_max + max(v_span * 0.25, 30))

                    ax_ins.tick_params(labelsize=6.5)
                    ax_ins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                    ax_ins.grid(True, linestyle=":", alpha=0.6)
                    ax_ins.legend(fontsize=7.5, loc="upper right")
                except Exception as e:
                    print(f"Warning: could not draw RF inset zoom: {e}")

        else:
            # Numeric axis (epochs)
            if not a2_sub.empty:
                x_a2 = [float(v) for v in a2_sub["value"]]
                ax.plot(x_a2, a2_sub["train_RMSE"], "--", color="#6baed6", linewidth=1.5, label="Default Train (a2)")
                ax.plot(x_a2, a2_sub["val_RMSE"], "--", color="#fd8d3c", linewidth=2.0, label="Default Val (a2)")
                best_a2 = int(a2_sub["val_RMSE"].idxmin())
                ax.axvline(x_a2[best_a2], color="#ff3c23", linestyle=":", alpha=0.8,
                           label=f"Best Default: ep {x_a2[best_a2]:.0f} ({a2_sub['val_RMSE'][best_a2]:,.0f})")

            if not b1_sub.empty:
                x_b1 = [float(v) for v in b1_sub["value"]]
                ax.plot(x_b1, b1_sub["train_RMSE"], "-", color="#08519c", linewidth=2.0, label="Tuned Train (b1)")
                ax.plot(x_b1, b1_sub["val_RMSE"], "-", color="#d94701", linewidth=2.5, label="Tuned Val (b1)")
                best_b1 = int(b1_sub["val_RMSE"].idxmin())
                ax.axvline(x_b1[best_b1], color="#2ca02c", linestyle="-.", alpha=0.9,
                           label=f"Best Tuned: ep {x_b1[best_b1]:.0f} ({b1_sub['val_RMSE'][best_b1]:,.0f})")

        ax.set_title(f"{meta['family']}: {meta['model']} - Complexity Comparison", fontsize=11, fontweight="bold")
        ax.set_xlabel(meta["xlabel"], fontsize=9)
        ax.set_ylabel("RMSE (MAD)", fontsize=9)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.grid(True, linestyle="--", alpha=0.5)
        if curve_key == "random_forest_max_depth":
            loc = "lower left"
            bbox = (0.0, 0.0)
        elif curve_key == "polynomial_degree":
            loc = "upper left"
            bbox = None
        elif curve_key == "knn_k":
            loc = "lower left"
            bbox = (0.03, 0.08)
        else:
            loc = "upper right"
            bbox = None

        ax.legend(fontsize=8, loc=loc, bbox_to_anchor=bbox, framealpha=0.85)

    fig.suptitle("Model Complexity Curves Comparison: Default (a2_preprocessed) vs Tuned (b1_tuned)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUT_DIR / "complexity_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig_path


def generate_complexity_comparison():
    df_a2 = load_stage_complexity("a2_preprocessed")
    df_b1 = load_stage_complexity("b1_tuned")

    if df_a2.empty or df_b1.empty:
        print("Warning: Missing complexity_curve.csv in results/a2_preprocessed/ or results/b1_tuned/.")
        return None, None

    fig_path = plot_comparison_panels(df_a2, df_b1)

    print("\nSaved complexity comparison outputs:")
    print(f"  - CSV (side-by-side): {OUT_DIR / 'complexity_comparison.csv'}")
    print(f"  - Plot (all models):  {fig_path}")


def main():
    generate_complexity_comparison()


if __name__ == "__main__":
    main()
