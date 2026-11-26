"""Compares the complexity curves of a2_preprocessed (default settings) and b1_tuned (tuned settings).
Run both complexity curve scripts first.

Output: results/comparison/complexity_comparison.png (4 panels, with zooms for Random Forest and Polynomial Regression)
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.shared import complexity as cc
from src.shared.core_utils import RESULTS_ROOT
from src.shared.plots import THOUSANDS, save_figure

OUT_DIR = RESULTS_ROOT / "comparison"

# order: x-axis order of the swept values; zoom_values: values shown in the inset zoom
CURVE_METADATA = {
    "polynomial_degree": {
        "family": "Linear",
        "model": "Polynomial Regression",
        "xlabel": "Polynomial degree (simple -> complex)",
        "categorical": True,
        "order": ["2", "3", "4", "5", "6", "7", "8"],
        "zoom_values": ["2", "3", "4"],
        "legend": ("upper left", None),
    },
    "random_forest_max_depth": {
        "family": "Tree",
        "model": "Random Forest",
        "xlabel": "Maximum tree depth (simple -> complex)",
        "categorical": True,
        "order": ["10", "20", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "None"],
        "zoom_values": ["40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "None"],
        "legend": ("lower left", (0.0, 0.0)),
    },
    "knn_k": {
        "family": "Distance",
        "model": "KNN",
        "xlabel": "Number of neighbours k (simple -> complex)",
        "categorical": True,
        "order": ["100", "75", "50", "30", "20", "15", "10", "9", "7", "5", "3", "1"],
        "legend": ("lower left", (0.03, 0.08)),
    },
    "dcnv2_epochs": {
        "family": "Neural",
        "model": "DCN-V2",
        "xlabel": "Training epochs (simple -> complex)",
        "categorical": False,
        "legend": ("upper right", None),
    },
}

# (column, line style, colour, label, line width, marker size); dashed = default (a2), solid = tuned (b1)
A2_LINES = [("train_RMSE", "o--", "#6baed6", "Default Train (a2)", 1.5, 4), ("val_RMSE", "o--", "#fd8d3c", "Default Val (a2)", 2.0, 5)]
B1_LINES = [("train_RMSE", "s-", "#08519c", "Tuned Train (b1)", 2.0, 5), ("val_RMSE", "s-", "#d94701", "Tuned Val (b1)", 2.5, 6)]
BILLIONS = plt.FuncFormatter(lambda v, _: f"{v / 1e9:,.0f} bn")


def column(data_map, values, name):
    """The column `name` at each x value (NaN where the stage has no row for it)."""
    return [data_map[v][name] if v in data_map else np.nan for v in values]


def best_value(sub):
    """(value, RMSE) of the lowest validation error."""
    best = int(sub["val_RMSE"].idxmin())
    return sub["value"][best], sub["val_RMSE"][best]


def rf_inset(ax, position, name, zx, a2_map, b1_map, best_a2, best_b1):
    """Random Forest zoom on the plateau depths: default vs tuned, for the training or validation RMSE."""
    kind = "Val" if name == "val_RMSE" else "Train"
    default_color, tuned_color = ("#fd8d3c", "#d94701") if kind == "Val" else ("#6baed6", "#08519c")
    ins = ax.inset_axes(position)
    a2, b1 = column(a2_map, zx, name), column(b1_map, zx, name)
    ins.plot(zx, a2, "o--", color=default_color, linewidth=1.8, markersize=4, label=f"Default {kind}")
    ins.plot(zx, b1, "s-", color=tuned_color, linewidth=2.0, markersize=5, label=f"Tuned {kind}")
    if best_b1 in zx:
        ins.axvline(best_b1, color="#2ca02c", linestyle="--", alpha=0.8)
    if best_a2 in zx:
        ins.axvline(best_a2, color="#fe3731", linestyle=":", alpha=0.8)
    valid = [v for v in a2 + b1 if not np.isnan(v)]
    margin = max((max(valid) - min(valid)) * 0.25, 30)
    ins.set_ylim(min(valid) - margin, max(valid) + margin)
    ins.tick_params(labelsize=6.5)
    ins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ins.grid(True, linestyle=":", alpha=0.6)
    ins.legend(fontsize=7.5, loc="center right")


def plot_epochs(ax, sub, lines, name, mark_color, best_style, best_alpha):
    """DCN-V2 epoch curve of one stage, ending where early stopping stops; marks the epoch whose weights are kept."""
    kept, stop = cc.early_stop_epochs(sub)
    run = sub.iloc[:stop]
    x = [float(v) for v in run["value"]]
    for column_name, style, color, label, width, _ in lines:
        ax.plot(x, run[column_name], style.lstrip("os"), color=color, linewidth=width, label=label)
    ax.plot([x[-1]] * 2, [run["train_RMSE"].iloc[-1], run["val_RMSE"].iloc[-1]], "x",
            color=mark_color, markersize=9, markeredgewidth=2.2, label=f"{name} stopped: ep {stop}")
    ax.axvline(kept, color=mark_color, linestyle=best_style, alpha=best_alpha, label=f"Best {name}: ep {kept}")
    (_, _, train_color, *_), (_, _, val_color, *_) = lines
    cc.label_point(ax, kept, sub["val_RMSE"][kept - 1], val_color, above=True)
    cc.label_point(ax, kept, sub["train_RMSE"][kept - 1], train_color, above=False)


def plot_comparison_panels(df_a2, df_b1):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, (curve_key, meta) in zip(axes.ravel(), CURVE_METADATA.items()):
        a2_sub = df_a2[df_a2["curve"] == curve_key].reset_index(drop=True)
        b1_sub = df_b1[df_b1["curve"] == curve_key].reset_index(drop=True)
        billions = False

        if meta["categorical"]:
            all_vals = list(dict.fromkeys(list(a2_sub["value"]) + list(b1_sub["value"])))
            x_vals = [v for v in meta["order"] if v in all_vals] + [v for v in all_vals if v not in meta["order"]]
            a2_map = {row["value"]: row for _, row in a2_sub.iterrows()}
            b1_map = {row["value"]: row for _, row in b1_sub.iterrows()}
            for data_map, lines in ((a2_map, A2_LINES), (b1_map, B1_LINES)):
                for name, style, color, label, width, size in lines:
                    ax.plot(x_vals, column(data_map, x_vals, name), style, color=color, linewidth=width, markersize=size, label=label)

            best_a2, score_a2 = best_value(a2_sub)
            best_b1, score_b1 = best_value(b1_sub)
            ax.axvline(best_a2, color="#ff3c23", linestyle=":", alpha=0.8, label=f"Best Default: {best_a2} ({score_a2:,.0f} MAD)")
            ax.axvline(best_b1, color="#2ca02c", linestyle="-.", alpha=0.9, label=f"Best Tuned: {best_b1} ({score_b1:,.0f} MAD)")

            if curve_key == "random_forest_max_depth":
                zx = [v for v in x_vals if v in meta["zoom_values"]]
                rf_inset(ax, [0.67, 0.68, 0.32, 0.3], "val_RMSE", zx, a2_map, b1_map, best_a2, best_b1)
                rf_inset(ax, [0.67, 0.2, 0.32, 0.3], "train_RMSE", zx, a2_map, b1_map, best_a2, best_b1)

            if curve_key == "polynomial_degree":
                values = [v for v in column(a2_map, x_vals, "val_RMSE") + column(b1_map, x_vals, "val_RMSE") if not np.isnan(v)]
                billions = max(values) > 10 * np.median(values)  # blow-up(s): big chart in billions, small chart zooms on degrees 2-4
                if billions:
                    zx = [v for v in x_vals if v in meta["zoom_values"]]
                    ins = ax.inset_axes([0.06, 0.1, 0.38, 0.42])
                    for data_map, lines in ((a2_map, A2_LINES), (b1_map, B1_LINES)):
                        for name, style, color, *_ in lines:
                            ins.plot(zx, column(data_map, zx, name), style, color=color, markersize=4)
                    ins.axvline(best_a2, color="#ff3c23", linestyle=":", alpha=0.8)
                    ins.axvline(best_b1, color="#2ca02c", linestyle="-.", alpha=0.9)
                    ins.set_title(f"Zoom: degree {zx[0]} to {zx[-1]} (MAD)", fontsize=8, pad=3)
                    ins.tick_params(labelsize=7)
                    ins.yaxis.set_major_formatter(THOUSANDS)
                    ins.grid(True, linestyle=":", alpha=0.6)
        else:
            plot_epochs(ax, a2_sub, A2_LINES, "Default", "#ff3c23", ":", 0.8)
            plot_epochs(ax, b1_sub, B1_LINES, "Tuned", "#2ca02c", "-.", 0.9)
            ax.set_xlim(0, None)

        ax.set_title(f"{meta['family']}: {meta['model']} - Complexity Comparison", fontsize=11, fontweight="bold")
        ax.set_xlabel(meta["xlabel"], fontsize=9)
        ax.set_ylabel("RMSE (MAD)", fontsize=9)
        ax.yaxis.set_major_formatter(BILLIONS if billions else THOUSANDS)
        ax.grid(True, linestyle="--", alpha=0.5)
        loc, bbox = meta["legend"]
        ax.legend(fontsize=8, loc=loc, bbox_to_anchor=bbox, framealpha=0.8)

    fig.suptitle("Model Complexity Curves Comparison: Default (a2_preprocessed) vs Tuned (b1_tuned)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUT_DIR / "complexity_comparison.png"
    save_figure(fig, fig_path)
    return fig_path


def main():
    df_a2 = cc.read_curve_csv(RESULTS_ROOT / "a2_preprocessed" / "complexity_curve.csv")
    df_b1 = cc.read_curve_csv(RESULTS_ROOT / "b1_tuned" / "complexity_curve.csv")
    print(f"Saved complexity comparison plot: {plot_comparison_panels(df_a2, df_b1)}")


if __name__ == "__main__":
    main()
