"""Shared code for the model complexity curves (a2_preprocessed_complexity_curve.py and b1_tuned_complexity_curve.py).

One setting that controls model complexity is swept from simple to complex. At each value the model is trained on the
training rows and scored (RMSE of the price in MAD) on those rows and on the validation rows. The test rows are never used.

Reading a panel: both errors high and close -> underfitting; training error keeps falling while validation error stops
falling or rises -> overfitting. Green dotted line = lowest validation error; grey dashed line (b1) = value picked by tuning.

Output: results/<stage>/complexity_curve.png and complexity_curve.csv (train_RMSE, val_RMSE, gap = val - train in MAD;
*_target columns are the RMSE on the model's own target scale).
"""
import math
import sys
from collections import namedtuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.preprocessing import full, minimal
from src.shared.core_utils import RANDOM_STATE, RESULTS_ROOT
from src.shared.plots import THOUSANDS
from src.shared.stages import load_data

POLY_DEGREES = [1, 2, 3, 4]
RF_DEPTHS = [2, 4, 6, 8, 12, 15, 22, 30, 45, 50, 60, 75, 100, None]  # None = fully grown; 50 = current tuned value
KNN_K = [100, 75, 50, 30, 20, 15, 10, 9, 7, 5, 3, 1]  # 9 = current tuned value
MAX_EPOCHS = 100

# One curve of a stage script.
#   name:        shown in the menu, e.g. "Random Forest (max_depth)"
#   title:       panel title
#   xlabel:      x-axis label
#   categorical: True = the swept values are drawn as evenly spaced labels (degree, depth, k); False = numeric axis (epochs)
#   tuned_value: value the tuning picked for the swept setting (grey dashed line), or None
#   compute:     function(split) -> table from sweep() or epoch_curve()
Curve = namedtuple("Curve", ["name", "title", "xlabel", "categorical", "tuned_value", "compute"])


def n_numeric_columns(stage):
    """Number of numeric columns (the last ones after encoding)."""
    csv_path, target = {"a1_baseline": (minimal.OUT_PATH, "Price")}.get(stage, (full.OUT_PATH, "Log Price"))
    return pd.read_csv(csv_path).drop(columns=target).select_dtypes(include="number").shape[1]


def polynomial(degree, n_numeric, **linreg_params):
    add_terms = ColumnTransformer([("numeric", PolynomialFeatures(degree=degree), list(range(-n_numeric, 0)))], remainder="passthrough")
    return Pipeline([("poly", add_terms), ("linreg", LinearRegression(**linreg_params))])


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(np.asarray(a).ravel(), np.asarray(b).ravel())))


def scores(split, value, train_pred, val_pred):
    price = split.inverse_transform_price
    row = {"value": value,
           "train_RMSE": _rmse(price(np.ravel(split.y_train)), price(np.ravel(train_pred))),
           "val_RMSE": _rmse(price(np.ravel(split.y_val)), price(np.ravel(val_pred))),
           "train_RMSE_target": _rmse(split.y_train, train_pred),
           "val_RMSE_target": _rmse(split.y_val, val_pred)}
    row["gap"] = row["val_RMSE"] - row["train_RMSE"]
    return row


def sweep(split, make_model, values):
    """sklearn models: fit one model per value."""
    rows = []
    for value in values:
        model = make_model(value).fit(split.X_train, split.y_train.ravel())
        rows.append(scores(split, value, model.predict(split.X_train), model.predict(split.X_val)))
        print(f"  {value}: train {rows[-1]['train_RMSE']:,.0f} | val {rows[-1]['val_RMSE']:,.0f}")
    return pd.DataFrame(rows)


def epoch_curve(split, model, optimizer, batch_size, predict, flat_target=False, clip_grad=None, max_epochs=MAX_EPOCHS):
    """PyTorch models: train once without early stopping and score after every epoch.
    flat_target: model outputs shape (n,) instead of (n, 1). clip_grad: max gradient norm."""
    device = next(model.parameters()).device
    torch.manual_seed(RANDOM_STATE)
    X_train = torch.as_tensor(split.X_train, device=device)
    y_train = torch.as_tensor(split.y_train.ravel() if flat_target else split.y_train, device=device)

    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), device=device)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            optimizer.zero_grad()
            F.mse_loss(model(X_train[batch]), y_train[batch]).backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        rows.append(scores(split, epoch, predict(model, split.X_train), predict(model, split.X_val)))
        if epoch % 10 == 0:
            print(f"  epoch {epoch}: train {rows[-1]['train_RMSE']:,.0f} | val {rows[-1]['val_RMSE']:,.0f}")
    return pd.DataFrame(rows)


# ---------- menu, CSV and figure ----------

def choose_curves(curves):
    """Ask which curves to (re)compute; returns a list of keys of `curves`."""
    keys = list(curves)
    print("Which curves do you want to (re)compute? The others keep their saved rows in complexity_curve.csv.")
    for number, key in enumerate(keys, start=1):
        print(f"  {number} = {curves[key].name}")
    print(f"  {len(keys) + 1} = all {len(keys)}")
    answer = input('Type one or more numbers separated by spaces (e.g. "1 3") and press Enter: ').split()
    if not answer or not all(a.isdigit() and 1 <= int(a) <= len(keys) + 1 for a in answer):
        sys.exit(f"'{' '.join(answer)}' is not an option.")
    if str(len(keys) + 1) in answer:
        return keys
    return [keys[int(a) - 1] for a in dict.fromkeys(answer)]  # keep the order typed, drop repeats


def _clean_value(value):
    """Swept values as text, the same whether just computed or read back from the CSV: 2 -> "2", 2.0 -> "2", None -> "None"."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value) in ("", "nan", "None"):
        return "None"
    text = str(value)
    return text.removesuffix(".0")


def update_csv(stage, new_tables):
    """Replace the rows of the recomputed curves in results/<stage>/complexity_curve.csv, keep the rest. Returns the full table."""
    path = RESULTS_ROOT / stage / "complexity_curve.csv"
    new = pd.concat(new_tables, names=["curve"]).reset_index(level=0).reset_index(drop=True)
    if path.exists():
        old = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
        old = old[~old["curve"].isin(new_tables)]
        new = pd.concat([old, new], ignore_index=True)
    new["value"] = new["value"].map(_clean_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def plot_panel(ax, table, title, xlabel, categorical=True, tuned_value=None):
    table = table.reset_index(drop=True)
    x = list(table["value"]) if categorical else [float(v) for v in table["value"]]
    style = "o-" if categorical else "-"
    ax.plot(x, table["train_RMSE"], style, color="#2a78d6", label="Training rows")
    ax.plot(x, table["val_RMSE"], style, color="#eda100", label="Validation rows")
    best = int(table["val_RMSE"].idxmin())
    ax.axvline(x[best], color="#2ca02c", linestyle=":",
               label=f"Lowest validation error: {table['value'][best]} ({table['val_RMSE'][best]:,.0f} MAD)")
    if tuned_value is not None:
        if categorical and _clean_value(tuned_value) not in x:
            print(f"  (tuned value {tuned_value} is not in the swept values {x}: add it to the list in complexity.py to see it marked)")
        else:
            ax.axvline(_clean_value(tuned_value) if categorical else tuned_value, color="grey", linestyle="--", label=f"Tuned value: {tuned_value}")
    ax.set(title=title, xlabel=xlabel, ylabel="RMSE (MAD)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)


def plot_all(stage, curves, table, suptitle):
    """One panel per curve, drawn from the full CSV table; a curve with no saved rows gets a 'not run yet' panel."""
    nrows = math.ceil(len(curves) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows), squeeze=False)
    for ax, (key, curve) in zip(axes.ravel(), curves.items()):
        rows = table[table["curve"] == key]
        if rows.empty:
            ax.set_title(curve.title)
            ax.text(0.5, 0.5, "Not run yet:\nrun the script and choose this curve", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            plot_panel(ax, rows, curve.title, curve.xlabel, curve.categorical, curve.tuned_value)
    for ax in axes.ravel()[len(curves):]:
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = RESULTS_ROOT / stage / "complexity_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(stage, curves, suptitle):
    """Menu -> compute the chosen curves -> update the CSV -> redraw the figure from the whole CSV."""
    chosen = choose_curves(curves)
    split = load_data(stage)
    new_tables = {}
    for key in chosen:
        print(f"\n{curves[key].name}")
        new_tables[key] = curves[key].compute(split)
    table = update_csv(stage, new_tables)
    plot_all(stage, curves, table, suptitle)
    print(f"\nUpdated {', '.join(chosen)} in complexity_curve.csv and redrew complexity_curve.png in {RESULTS_ROOT / stage}")
