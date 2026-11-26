"""Shared code for the complexity curves (a2_preprocessed_complexity_curve.py, b1_tuned_complexity_curve.py).

One complexity setting is swept from simple to complex; at each value the model is scored (RMSE in MAD) on the training
and validation rows, never the test rows. Both errors high -> underfitting; validation error rising while training error
falls -> overfitting. Output: results/<stage>/complexity_curve.png and .csv.
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
from sklearn.metrics import mean_squared_error

from src.shared.core_utils import RANDOM_STATE, RESULTS_ROOT
from src.shared.plots import THOUSANDS, save_figure
from src.shared.stages import load_data

POLY_DEGREES = [2, 3, 4, 5, 6, 7, 8]
RF_DEPTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]  # None = fully grown; 50 = current tuned value
KNN_K = [100, 75, 50, 30, 20, 15, 10, 9, 7, 5, 3, 1]  # 9 = current tuned value
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping patience of the real training (run_neural.PATIENCE, train_torch(patience=10))

# One curve of a stage script.
#   name:        shown in the menu, e.g. "Random Forest (max_depth)"
#   title:       panel title
#   xlabel:      x-axis label
#   categorical: True = the swept values are drawn as evenly spaced labels (degree, depth, k); False = numeric axis (epochs)
#   tuned_value: value the tuning picked for the swept setting (grey dashed line), or None
#   compute:     function(split) -> table from sweep() or epoch_curve()
#   zoom_range:  optional (low, high) range of values for the validation zoom, high may be None (e.g. (40, 100))
Curve = namedtuple("Curve", ["name", "title", "xlabel", "categorical", "tuned_value", "compute", "zoom_range"], defaults=[None])


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


def early_stop_epochs(table, patience=PATIENCE):
    """Replay early stopping on an epoch curve, the same rule as the real training: stop once the validation loss on the
    model's target scale (val_RMSE_target) has not improved for `patience` epochs, keep the weights of the best epoch.
    Returns (epoch whose weights are kept, epoch where training stops). Works on a full 100-epoch table too."""
    best, kept, bad = float("inf"), 1, 0
    for epoch, loss in enumerate(table["val_RMSE_target"].astype(float), start=1):
        if loss < best:
            best, kept, bad = loss, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                return kept, epoch
    return kept, len(table)


def epoch_curve(split, model, optimizer, batch_size, predict, flat_target=False, clip_grad=None, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """PyTorch models: train once with early stopping and score after every epoch. Training stops once the validation
    loss (target scale) has not improved for `patience` epochs, like the real training; patience=None trains all max_epochs.
    flat_target: model outputs shape (n,) instead of (n, 1). clip_grad: max gradient norm."""
    device = next(model.parameters()).device
    torch.manual_seed(RANDOM_STATE)
    X_train = torch.as_tensor(split.X_train, device=device)
    y_train = torch.as_tensor(split.y_train.ravel() if flat_target else split.y_train, device=device)

    rows, best_loss, kept = [], float("inf"), 0
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
        if patience:
            if rows[-1]["val_RMSE_target"] < best_loss:
                best_loss, kept = rows[-1]["val_RMSE_target"], epoch
            elif epoch - kept >= patience:
                print(f"  early stopping at epoch {epoch} (keeps epoch {kept})")
                break
    return pd.DataFrame(rows)


# ---------- menu, CSV and figure ----------

def choose_curves(curves):
    """Ask which curves to (re)compute; returns a list of keys of `curves`."""
    if "--redraw" in sys.argv:
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
    return str(value).removesuffix(".0")


def read_curve_csv(path):
    """A complexity_curve.csv with the swept values as clean text."""
    table = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
    table["value"] = table["value"].map(_clean_value)
    return table


def update_csv(stage, new_tables):
    """Replace the rows of the recomputed curves in results/<stage>/complexity_curve.csv, keep the rest. Returns the full table."""
    path = RESULTS_ROOT / stage / "complexity_curve.csv"
    if not new_tables:
        return read_curve_csv(path)
    new = pd.concat(new_tables, names=["curve"]).reset_index(level=0).reset_index(drop=True)
    if path.exists():
        old = read_curve_csv(path)
        old = old[~old["curve"].isin(new_tables)]
        new = pd.concat([old, new], ignore_index=True)
    new["value"] = new["value"].map(_clean_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def filter_zoom_table(table, zoom_range=None):
    """Rows shown in the validation zoom: values inside zoom_range = (low, high), high None = no upper limit
    (then "None", a fully grown tree, is inside too). Without a range, depths 10 and 20 (early underfitting)
    are left out when the plateau values are there."""
    values = table["value"].map(_clean_value)
    if zoom_range is None:
        if values.isin(["10", "20"]).any() and values.isin(["30", "40", "50", "60", "70"]).any():
            sub = table[~values.isin(["10", "20"])]
        else:
            sub = table
    else:
        low, high = zoom_range
        sub = table[[high is None if v == "None" else low <= float(v) and (high is None or float(v) <= high) for v in values]]
    return (sub if not sub.empty else table).reset_index(drop=True)


def label_point(ax, x, y, color, above=True):
    """Dot on the curve with its RMSE written next to it (above or below)."""
    ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="black", markeredgewidth=0.8, zorder=6)
    ax.annotate(f"{y:,.0f}", (x, y), xytext=(0, 11 if above else -11), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=9, fontweight="bold", color=color, zorder=7,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "lw": 1, "alpha": 0.95})


def plot_panel(ax, table, title, xlabel, categorical=True, tuned_value=None, zoom_range=None):
    table = table.reset_index(drop=True)
    if not categorical and "val_RMSE_target" in table:
        # epoch curve: draw it only up to where early stopping ends training; mark the epoch whose weights are kept
        kept, stop = early_stop_epochs(table)
        table = table.iloc[:stop]
    x = list(table["value"]) if categorical else [float(v) for v in table["value"]]
    style = "o-" if categorical else "-"
    ax.plot(x, table["train_RMSE"], style, color="#2a78d6", label="Training rows")
    ax.plot(x, table["val_RMSE"], style, color="#eda100", label="Validation rows")
    if not categorical and "val_RMSE_target" in table:
        best = kept - 1
        ax.plot([x[-1]] * 2, [table["train_RMSE"].iloc[-1], table["val_RMSE"].iloc[-1]], "x", color="#d62728",
                markersize=9, markeredgewidth=2.2, label=f"Early stopping ends training: epoch {stop}")
        ax.axvline(x[best], color="#2ca02c", linestyle=":", label=f"Kept weights (lowest validation loss): epoch {kept}")
        label_point(ax, x[best], table["val_RMSE"][best], "#eda100", above=True)
        label_point(ax, x[best], table["train_RMSE"][best], "#2a78d6", above=False)
        ax.set_xlim(0, None)
    else:
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

    # Inset zoom where the validation curve is nearly flat next to the train-val range (e.g. Random Forest)
    val_span = float(table["val_RMSE"].max() - table["val_RMSE"].min())
    total_span = float(max(table["val_RMSE"].max(), table["train_RMSE"].max()) - min(table["val_RMSE"].min(), table["train_RMSE"].min()))
    has_inset = total_span > 0 and (val_span / total_span < 0.05 or "Random Forest" in title)

    if has_inset:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        zoom_table = filter_zoom_table(table, zoom_range)
        zx = list(zoom_table["value"]) if categorical else [float(v) for v in zoom_table["value"]]
        z_best = int(zoom_table["val_RMSE"].idxmin())
        y_margin = max(float(zoom_table["val_RMSE"].max() - zoom_table["val_RMSE"].min()) * 0.25, 20)

        ax_ins = ax.inset_axes([0.68, 0.22, 0.31, 0.3])
        ax_ins.plot(zx, zoom_table["val_RMSE"], "o-", color="#eda100", markersize=4, linewidth=1.5)
        ax_ins.axvline(zx[z_best], color="#2ca02c", linestyle=":", alpha=0.8)
        ax_ins.plot(zx[z_best], zoom_table["val_RMSE"][z_best], "o", color="#2ca02c", markersize=5)
        ax_ins.set_ylim(zoom_table["val_RMSE"].min() - y_margin, zoom_table["val_RMSE"].max() + y_margin)
        if len(zoom_table) < len(table):
            ax_ins.set_title("Val Zoom", fontsize=7.5, pad=3)
        else:
            ax_ins.set_title("Validation Zoom (MAD)", fontsize=8, pad=3)
        ax_ins.tick_params(labelsize=7)
        ax_ins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_ins.grid(True, linestyle=":", alpha=0.6)
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
            plot_panel(ax, rows, curve.title, curve.xlabel, curve.categorical, curve.tuned_value, curve.zoom_range)
    for ax in axes.ravel()[len(curves):]:
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, RESULTS_ROOT / stage / "complexity_curve.png")


def run(stage, curves, suptitle):
    """Menu -> compute the chosen curves -> update the CSV -> redraw the figure from the whole CSV."""
    chosen = choose_curves(curves)
    new_tables = {}
    if chosen:
        split = load_data(stage)
        for key in chosen:
            print(f"\n{curves[key].name}")
            new_tables[key] = curves[key].compute(split)
    table = update_csv(stage, new_tables)
    plot_all(stage, curves, table, suptitle)
    action = f"Updated {', '.join(chosen)} in complexity_curve.csv and redrew" if chosen else "Redrew"
    print(f"\n{action} complexity_curve.png in {RESULTS_ROOT / stage}")
