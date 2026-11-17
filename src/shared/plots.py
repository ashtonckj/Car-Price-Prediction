"""Graphs for the model files (one family) and comparison.py (all families / stages)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; no window pops up
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

THOUSANDS = FuncFormatter(lambda value, _: f"{value / 1000:,.0f}k" if value else "0")  # 250000 -> "250k"
COLORS = ["#2a78d6", "#800080", "#eda100", "#2ca02c"]  # by position: 1st model, 2nd model, ...


def plot_family(title, y_true_price, predicted_prices, models, save_path):
    """Actual vs predicted price, one panel per model. predicted_prices: {name: prices in MAD}; models: {name: {"metrics": ...}}.
    Plain (linear) axes in MAD, labelled 100k, 200k, ...; predictions outside the axis range are drawn at its edge."""
    hi = float(np.max(y_true_price)) * 1.05

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5), squeeze=False)
    for i, (ax, (name, r)) in enumerate(zip(axes[0], models.items())):
        ax.scatter(y_true_price, np.clip(predicted_prices[name], 0, hi), s=6, alpha=0.3, color=COLORS[i % len(COLORS)], linewidths=0, rasterized=True)
        ax.plot([0, hi], [0, hi], "r--", linewidth=1)
        ax.set(xlim=(0, hi), ylim=(0, hi), aspect="equal")
        ax.xaxis.set_major_formatter(THOUSANDS)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_xlabel("Actual price (MAD)")
        ax.set_ylabel("Predicted price (MAD)")
        ax.set_title(f"{name}\nR2 {r['metrics']['R2']:.4f} | MAE {r['metrics']['MAE']:,.0f} MAD", fontsize=11)
    fig.suptitle(f"{title}: actual vs predicted price (test set)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_bars(table, metric, title, save_path):
    """Horizontal bars of one metric for every model of one stage (error bars = CV std when available)."""
    higher_is_better = metric == "R2"
    table = table.sort_values(metric, ascending=not higher_is_better)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
    errors = table.get(f"CV_{metric}_std")
    ax.barh(table["Family"] + ": " + table["Model"], table[metric], xerr=errors, capsize=3, color=COLORS[0])
    ax.set_xlabel(f"{metric} (test set; error bars = CV std)" if errors is not None else metric)
    ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stage_comparison(pivot, metric, save_path):
    """Grouped bars: one group per model, one bar per stage (`pivot` = models x stages)."""
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(pivot) + 3), 5.5))
    width = 0.8 / len(pivot.columns)
    x = np.arange(len(pivot))
    for i, stage in enumerate(pivot.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[stage], width, label=stage, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x, pivot.index, rotation=35, ha="right")
    ax.set_ylabel(metric)
    if metric == "R2":
        # R2 is at most 1; very negative values are cut at -1 so they don't hide the other bars
        ax.set_ylim(max(-1, min(0, np.nanmin(pivot.to_numpy()))), 1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{metric} on the test set, by stage")
    ax.legend()
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
