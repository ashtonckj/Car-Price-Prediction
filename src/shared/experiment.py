"""Shared helpers: timing, metrics, saving results (JSON + graph), and an all-model comparison.

Running this file directly compares all saved models:
    python src/shared/experiment.py
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "default_runs"
COLORS = ["#2a78d6", "#800080", "#eda100"]


def timed(function, *args, **kwargs):
    """Call function(*args, **kwargs) and return (its result, seconds it took)."""
    t_start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - t_start


def compute_metrics(y_true_scaled, y_pred_scaled, inverse_transform_price):
    y_true, y_pred = inverse_transform_price(y_true_scaled), inverse_transform_price(y_pred_scaled)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2_log": float(r2_score(y_true_scaled, y_pred_scaled)),
    }


def plot_family(title, y_true_price, models, save_path):
    """Actual vs predicted price, one panel per model (log axes, same limits in every panel)."""
    lo, hi = np.percentile(y_true_price, [0.1, 99.9])
    lo, hi = max(lo, 1000) / 1.5, hi * 1.5

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5), squeeze=False)
    for i, (ax, (name, r)) in enumerate(zip(axes[0], models.items())):
        ax.scatter(y_true_price, np.clip(r["y_pred_price"], lo, hi), s=6, alpha=0.3,
                   color=COLORS[i % len(COLORS)], linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel("Actual price (MAD, log scale)")
        ax.set_ylabel("Predicted price (MAD, log scale)")
        ax.set_title(f"{name}\nR2 {r['metrics']['R2']:.4f} | R2 log {r['metrics']['R2_log']:.4f}", fontsize=11)
    fig.suptitle(f"{title}: actual vs predicted price (test set)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_and_plot(key, title, results, y_test, inverse_transform_price):
    """Write <key>_results.json (family, metrics, times) and <key>_pred_vs_actual.png."""
    models = {}
    for name, r in results.items():
        y_pred = np.asarray(r["y_pred"]).reshape(-1)  # sklearn gives (n,), torch gives (n, 1)
        models[name] = {
            "metrics": compute_metrics(y_test, y_pred, inverse_transform_price),
            "fit_time_s": r["fit_time"],
            "predict_time_s": r["predict_time"],
            "y_pred_price": inverse_transform_price(y_pred),
        }
    saved = {
        "family": title,
        "models": {
            name: {k: v for k, v in m.items() if k != "y_pred_price"}
            for name, m in models.items()
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_family(title, inverse_transform_price(y_test), models, RESULTS_DIR / f"{key}_pred_vs_actual.png")
    (RESULTS_DIR / f"{key}_results.json").write_text(json.dumps(saved, indent=2))
    print(f"\nSaved {key}_results.json and {key}_pred_vs_actual.png in {RESULTS_DIR}")


def results_table():
    """One row per model across all saved families, best log-space R2 first."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("*_results.json")):
        fam = json.loads(p.read_text())
        for name, r in fam["models"].items():
            rows.append({"Family": fam["family"], "Model": name, **r["metrics"],
                         "Fit_time_s": r["fit_time_s"], "Predict_time_s": r["predict_time_s"]})
    return pd.DataFrame(rows).sort_values("R2_log", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def plot_comparison(table, metric="R2_log", save_path=None):
    """Horizontal bar chart of one metric for every model."""
    table = table.sort_values(metric, ascending=metric not in ("R2", "R2_log"))
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 1.5))
    ax.barh(table["Model"], table[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"All models: {metric} (test set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    table = results_table()
    if table.empty:
        print(f"No *_results.json found in {RESULTS_DIR}")
    else:
        plot_comparison(table, "R2_log", RESULTS_DIR / "comparison_R2_log.png")
        print(table.to_string(index=False, float_format="{:,.4f}".format))