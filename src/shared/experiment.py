"""Shared helpers: timing, metrics, saving results as JSON, and graphs/comparisons from the saved JSON.

Running this file directly redraws every family graph from the saved JSON and compares all models:
    python src/shared/experiment.py
"""
import json
import time
from datetime import datetime
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

# Colours by position in the family: 1st model, 2nd model, 3rd model (repeats if a family has more)
COLORS = ["#2a78d6", "#eb6834", "#eda100"]


def timed(function, *args, **kwargs):
    """Call function(*args, **kwargs) and return (its result, seconds it took)."""
    t0 = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - t0


def compute_metrics(y_true_scaled, y_pred_scaled, inverse_transform_price):
    """Same numbers as evaluate(), returned as a dict (so they can be saved)."""
    y_true, y_pred = inverse_transform_price(y_true_scaled), inverse_transform_price(y_pred_scaled)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2_log": float(r2_score(y_true_scaled, y_pred_scaled)),
    }


# ---------------------------------------------------------------------------
# Saving (called at the end of every family file)
# ---------------------------------------------------------------------------
def save_and_plot(key, title, results, y_test, inverse_transform_price):
    """Write <key>_results.json (metrics, times, test predictions in MAD) and <key>_pred_vs_actual.png.

    results: {name: {"model", "y_pred", "fit_time", "predict_time"}} as built by a family file.
    """
    models = {}
    for name, r in results.items():
        y_pred = np.asarray(r["y_pred"]).reshape(-1)  # sklearn gives (n,), torch gives (n, 1)
        models[name] = {
            "metrics": compute_metrics(y_test, y_pred, inverse_transform_price),
            "fit_time_s": r["fit_time"],
            "predict_time_s": r["predict_time"],
            "y_pred_price": inverse_transform_price(y_pred).tolist(),
        }
    saved = {
        "family": title,
        "created": datetime.now().isoformat(timespec="seconds"),
        "y_true_price": inverse_transform_price(y_test).tolist(),
        "models": models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{key}_results.json").write_text(json.dumps(saved))
    plot_family(saved, RESULTS_DIR / f"{key}_pred_vs_actual.png")
    print(f"\nSaved {RESULTS_DIR / f'{key}_results.json'} and {key}_pred_vs_actual.png")


# ---------------------------------------------------------------------------
# Loading, comparisons and graphs (work from the saved JSON, no model re-runs)
# ---------------------------------------------------------------------------
def load_results(results_dir=RESULTS_DIR):
    """Load every *_results.json in a folder -> {family_key: saved dict}."""
    return {p.stem.removesuffix("_results"): json.loads(p.read_text())
            for p in sorted(Path(results_dir).glob("*_results.json"))}


def results_table(all_results):
    """One row per model across all loaded families, best log-space R2 first."""
    rows = [{"Family": fam["family"], "Model": name, **r["metrics"],
             "Fit_time_s": r["fit_time_s"], "Predict_time_s": r["predict_time_s"]}
            for fam in all_results.values() for name, r in fam["models"].items()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("R2_log", ascending=False).reset_index(drop=True)


def plot_family(family_results, save_path=None):
    """Actual vs predicted price, one panel per model side by side (log axes, same limits in every panel)."""
    true_price = np.array(family_results["y_true_price"])
    lo, hi = np.percentile(true_price, [0.1, 99.9])
    lo, hi = max(lo, 1000) / 1.5, hi * 1.5

    models = family_results["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5), squeeze=False)
    for i, (ax, (name, r)) in enumerate(zip(axes[0], models.items())):
        pred = np.array(r["y_pred_price"])
        ax.scatter(true_price, np.clip(pred, lo, hi), s=6, alpha=0.3, color=COLORS[i % len(COLORS)], linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel("Actual price (MAD, log scale)")
        ax.set_ylabel("Predicted price (MAD, log scale)")
        ax.set_title(f"{name}\nR2 {r['metrics']['R2']:.4f} | R2 log {r['metrics']['R2_log']:.4f}", fontsize=11)
    fig.suptitle(f"{family_results['family']}: actual vs predicted price (test set)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def plot_comparison(table, metric="R2_log", save_path=None):
    """Horizontal bar chart of one metric (any column of results_table) for every model."""
    table = table.sort_values(metric, ascending=metric not in ("R2", "R2_log"))
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 1.5))
    ax.barh(table["Model"], table[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"All models: {metric} (test set)")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    saved = load_results()
    for key, family in saved.items():
        plot_family(family, RESULTS_DIR / f"{key}_pred_vs_actual.png")
    table = results_table(saved)
    if table.empty:
        print(f"No *_results.json found in {RESULTS_DIR}")
    else:
        plot_comparison(table, "R2_log", RESULTS_DIR / "comparison_R2_log.png")
        print(table.to_string(index=False, float_format="{:,.4f}".format))