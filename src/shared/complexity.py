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
RF_DEPTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]  # None = fully grown; 50 = current tuned value
KNN_K = [100, 75, 50, 30, 20, 15, 10, 9, 7, 5, 3, 1]  # 9 = current tuned value
MAX_EPOCHS = 100

# One curve of a stage script.
#   name:        shown in the menu, e.g. "Random Forest (max_depth)"
#   title:       panel title
#   xlabel:      x-axis label
#   categorical: True = the swept values are drawn as evenly spaced labels (degree, depth, k); False = numeric axis (epochs)
#   tuned_value: value the tuning picked for the swept setting (grey dashed line), or None
#   compute:     function(split) -> table from sweep() or epoch_curve()
#   zoom_range:  optional range of values to focus the validation zoom on (e.g. (30, 70) or ["30", "40", "50", "60", "70"])
Curve = namedtuple("Curve", ["name", "title", "xlabel", "categorical", "tuned_value", "compute", "zoom_range"], defaults=[None])


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
    if any(arg in sys.argv for arg in ("--redraw", "--plot-only", "--rf-only")):
        return []
    keys = list(curves)
    print("Which curves do you want to (re)compute? The others keep their saved rows in complexity_curve.csv.")
    print("  0 = Redraw existing plots from CSV (no recomputation)")
    for number, key in enumerate(keys, start=1):
        print(f"  {number} = {curves[key].name}")
    print(f"  {len(keys) + 1} = all {len(keys)}")
    answer = input('Type one or more numbers separated by spaces (e.g. "1 3") and press Enter: ').split()
    if not answer or not all(a.isdigit() and 0 <= int(a) <= len(keys) + 1 for a in answer):
        sys.exit(f"'{' '.join(answer)}' is not an option.")
    if "0" in answer:
        return []
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
    if not new_tables:
        if path.exists():
            old = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
            old["value"] = old["value"].map(_clean_value)
            return old
        return pd.DataFrame()
    new = pd.concat(new_tables, names=["curve"]).reset_index(level=0).reset_index(drop=True)
    if path.exists():
        old = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
        old = old[~old["curve"].isin(new_tables)]
        new = pd.concat([old, new], ignore_index=True)
    new["value"] = new["value"].map(_clean_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def get_cli_zoom_range():
    """Extract optional --zoom-range or --depth-range from command-line arguments."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--zoom-range", "--depth-range") and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            for sep in (",", "..", "-", ":"):
                if sep in val:
                    parts = [p.strip() for p in val.split(sep)]
                    if len(parts) == 2 and sep != ",":
                        low = parts[0] if parts[0] not in ("", "None") else None
                        high = parts[1] if parts[1] not in ("", "None") else None
                        return (low, high)
                    return parts
            return [val.strip()]
    return None


def filter_zoom_table(table, zoom_range=None):
    """Filter table rows for the zoomed-in validation view based on specified depth/x range."""
    cli_range = get_cli_zoom_range()
    effective_range = cli_range if cli_range is not None else zoom_range

    val_clean = table["value"].map(_clean_value)

    if effective_range is None:
        # Default for Random Forest: if early underfitting depths (10, 20) are present,
        # exclude them to zoom into the plateau/minimum region (depths >= 30)
        has_small = any(v in ("10", "20") for v in val_clean)
        has_plateau = any(v in ("30", "40", "50", "60", "70") for v in val_clean)
        if has_small and has_plateau:
            sub = table[~val_clean.isin(["10", "20"])].reset_index(drop=True)
            return sub if not sub.empty else table.reset_index(drop=True)
        return table.reset_index(drop=True)

    # List or set of explicit target values: e.g. ["30", "40", "50", "60", "70"]
    if isinstance(effective_range, (list, set)):
        targets = [_clean_value(v) for v in effective_range]
        sub = table[val_clean.isin(targets)].reset_index(drop=True)
        return sub if not sub.empty else table.reset_index(drop=True)

    # Tuple of (min_depth, max_depth): e.g. (30, 70) or (30, None)
    if isinstance(effective_range, tuple) and len(effective_range) == 2:
        low, high = effective_range
        low_clean = _clean_value(low) if low is not None else None
        high_clean = _clean_value(high) if high is not None else None

        def matches(v):
            c = _clean_value(v)
            if c == "None":
                return high is None or high_clean == "None"
            try:
                num = float(c)
                if low is not None and low_clean != "None" and num < float(low_clean):
                    return False
                return not (high is not None and high_clean != "None" and num > float(high_clean))
            except ValueError:
                return True

        mask = [matches(v) for v in table["value"]]
        sub = table[mask].reset_index(drop=True)
        return sub if not sub.empty else table.reset_index(drop=True)

    return table.reset_index(drop=True)


def plot_panel(ax, table, title, xlabel, categorical=True, tuned_value=None, zoom_range=None):
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

    # Inset zoom for curves where validation variation is small relative to train-val range (e.g. Random Forest)
    val_span = float(table["val_RMSE"].max() - table["val_RMSE"].min())
    total_span = float(max(table["val_RMSE"].max(), table["train_RMSE"].max()) - min(table["val_RMSE"].min(), table["train_RMSE"].min()))
    has_inset = total_span > 0 and (val_span / total_span < 0.05 or "Random Forest" in title)

    if has_inset:
        # Position legend on the middle-left so it doesn't collide with the inset zoom on the right
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        try:
            zoom_table = filter_zoom_table(table, zoom_range)
            zx = list(zoom_table["value"]) if categorical else [float(v) for v in zoom_table["value"]]
            z_best = int(zoom_table["val_RMSE"].idxmin())
            z_val_span = float(zoom_table["val_RMSE"].max() - zoom_table["val_RMSE"].min())

            ax_ins = ax.inset_axes([0.68, 0.22, 0.31, 0.3])
            ax_ins.plot(zx, zoom_table["val_RMSE"], "o-", color="#eda100", markersize=4, linewidth=1.5)
            ax_ins.axvline(zx[z_best], color="#2ca02c", linestyle=":", alpha=0.8)
            ax_ins.plot(zx[z_best], zoom_table["val_RMSE"][z_best], "o", color="#2ca02c", markersize=5)
            y_margin = max(z_val_span * 0.25, 20)
            ax_ins.set_ylim(zoom_table["val_RMSE"].min() - y_margin, zoom_table["val_RMSE"].max() + y_margin)
            
            if len(zoom_table) < len(table):
                ax_ins.set_title("Val Zoom", fontsize=7.5, pad=3)
            else:
                ax_ins.set_title("Validation Zoom (MAD)", fontsize=8, pad=3)
            ax_ins.tick_params(labelsize=7)
            ax_ins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax_ins.grid(True, linestyle=":", alpha=0.6)
        except Exception:
            pass
    else:
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
            zoom_range = getattr(curve, "zoom_range", None)
            plot_panel(ax, rows, curve.title, curve.xlabel, curve.categorical, curve.tuned_value, zoom_range=zoom_range)
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
    if chosen:
        split = load_data(stage)
        new_tables = {}
        for key in chosen:
            print(f"\n{curves[key].name}")
            new_tables[key] = curves[key].compute(split)
        table = update_csv(stage, new_tables)
    else:
        table = update_csv(stage, {})
    plot_all(stage, curves, table, suptitle)
    action = f"Updated {', '.join(chosen)} in complexity_curve.csv and redrew" if chosen else "Redrew"
    print(f"\n{action} complexity_curve.png in {RESULTS_ROOT / stage}")
