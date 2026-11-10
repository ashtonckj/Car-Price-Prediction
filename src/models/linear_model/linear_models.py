import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

matplotlib.use("Agg")  
print("Starting linear_models.py - loading libraries...", flush=True)


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_data

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = {
    "Linear Regression": sklearn.linear_model.LinearRegression(),
    "Ridge": sklearn.linear_model.Ridge(random_state=RANDOM_STATE),
    "Lasso": sklearn.linear_model.Lasso(random_state=RANDOM_STATE),
    "Huber": sklearn.linear_model.HuberRegressor(),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def price_metrics(y_true, y_pred, inverse_transform_price) -> dict:
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    true_price = inverse_transform_price(y_true)
    pred_price = inverse_transform_price(y_pred)
    return {
        "RMSE": np.sqrt(mean_squared_error(true_price, pred_price)),
        "MAE": mean_absolute_error(true_price, pred_price),
        "R2": r2_score(true_price, pred_price),
        "MAPE": mean_absolute_percentage_error(true_price, pred_price) * 100,
        "R2_log": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Training and testing
# ---------------------------------------------------------------------------
def train_and_test(name, model, data):
    X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = data

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.time()
        model.fit(X_train, y_train.ravel())
        fit_time = time.time() - t0
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print(f"Note: {name} stopped at its default iteration limit before fully converging.")
    print(f"Fit time (s): {fit_time:.2f}")

    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0
    print(f"Predict time (s): {predict_time:.2f}")

    evaluate(name, y_test, y_pred, inverse_transform_price)

    row = {"Model": name, **price_metrics(y_test, y_pred, inverse_transform_price),
           "Fit_time_s": fit_time, "Predict_time_s": predict_time}
    return row, inverse_transform_price(y_pred)


# ---------------------------------------------------------------------------
# Charts (for the report)
# ---------------------------------------------------------------------------
MODEL_COLORS = {"Linear Regression": "#2a78d6", "Ridge": "#eb6834",
                "Lasso": "#1baf7a", "Huber": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [
    ("RMSE", "RMSE (MAD) - lower is better", "{:,.0f}"),
    ("MAE", "MAE (MAD) - lower is better", "{:,.0f}"),
    ("MAPE", "MAPE (%) - lower is better", "{:.2f}%"),
    ("R2", "R² (MAD scale) - higher is better", "{:.4f}"),
    ("R2_log", "R² (log space) - higher is better", "{:.4f}"),
    ("Fit_time_s", "Fit time (s) - lower is better", "{:.2f}s"),
]


def _style_axis(ax):
    """Light grid, no box, grey tick labels."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)


def plot_metric_bars(results: pd.DataFrame, path: Path):
    """Six panels (one per metric), one bar per model."""
    results = results.set_index("Model")
    names = list(results.index)
    colors = [MODEL_COLORS.get(n, MUTED) for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, panel_title, fmt) in zip(axes.flat, PANELS):
        values = results[col].to_numpy()
        ax.bar(x, values, width=0.6, color=colors, edgecolor="white", linewidth=2)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        span = max(values.max(), 0) - min(values.min(), 0)
        for xi, v in zip(x, values):
            # Label above positive bars and below negative ones (e.g. Lasso's R²)
            ax.text(xi, v + (0.02 if v >= 0 else -0.02) * span, fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
        ax.set_ylim(min(values.min(), 0) - 0.12 * span * (values.min() < 0),
                    max(values.max(), 0) + 0.15 * span)
        if col in ("RMSE", "MAE"):  # 200,000 instead of 200000 on the axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks(x, names, rotation=15)
        ax.set_title(panel_title, fontsize=11, color=INK, loc="left")
        _style_axis(ax)
    fig.suptitle("Linear family (default settings): performance on the test set (20%)",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pred_vs_actual(true_price, predictions: dict, path: Path):
    names = list(predictions)
    lo, hi = np.percentile(true_price, [0.1, 99.9])
    lo, hi = max(lo, 1000) / 1.5, hi * 1.5
    n_rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5.5 * n_rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        pred = np.clip(predictions[name], lo, hi)  
        ax.scatter(true_price, pred, s=4, alpha=0.2, color=MODEL_COLORS.get(name, MUTED),
                   linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "--", color=MUTED, linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel("Actual price (MAD, log scale)", color=MUTED)
        ax.set_ylabel("Predicted price (MAD, log scale)", color=MUTED)
        ax.set_title(name, fontsize=12, color=INK, loc="left")
        _style_axis(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Linear family: predicted vs actual price on the test set",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    data = load_data()
    y_test, inverse_transform_price = data[5], data[6]

    rows, predictions = [], {}
    for name, model in MODELS.items():
        print(f"\n===== {name} =====")
        try:
            row, pred_price = train_and_test(name, model, data)
            rows.append(row)
            predictions[name] = pred_price
        except Exception as err:
            print(f"FAILED {name}: {err}")
    if not rows:
        print("Every model failed - nothing to save.")
        return
    results = pd.DataFrame(rows)

    print("\n--- Linear family: test set summary ---")
    print(results.to_string(index=False, formatters={
        "RMSE": "{:,.0f}".format, "MAE": "{:,.0f}".format, "R2": "{:.4f}".format,
        "MAPE": "{:.2f}".format, "R2_log": "{:.4f}".format,
        "Fit_time_s": "{:.2f}".format, "Predict_time_s": "{:.3f}".format}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "results_linear.csv", index=False)
    charts = [
        ("linear_test_metrics.png", lambda p: plot_metric_bars(results, p)),
        ("linear_pred_vs_actual.png",
         lambda p: plot_pred_vs_actual(inverse_transform_price(y_test), predictions, p)),
    ]
    for filename, make_chart in charts:
        try:  
            make_chart(RESULTS_DIR / filename)
            print(f"Saved chart: {filename}")
        except Exception as err:
            print(f"FAILED to draw {filename}: {err}")
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()