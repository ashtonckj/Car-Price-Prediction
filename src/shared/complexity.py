"""Shared code for the model complexity curves (a2_preprocessed_complexity_curve.py and b1_tuned_complexity_curve.py).

One setting that controls model complexity is swept from simple to complex. At each value the model is trained on the
training rows and scored (RMSE of the price in MAD) on those rows and on the validation rows. The test rows are never used.

Reading a panel: both errors high and close -> underfitting; training error keeps falling while validation error stops
falling or rises -> overfitting. Green dotted line = lowest validation error; grey dashed line (b1) = value picked by tuning.

Output: results/<stage>/complexity_curve.png and complexity_curve.csv (train_RMSE, val_RMSE, gap = val - train in MAD;
*_target columns are the RMSE on the model's own target scale).
"""
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

POLY_DEGREES = [1, 2, 3, 4]
RF_DEPTHS = [2, 4, 6, 8, 12, 15, 22, 30, 45, 60, 75, 100, None]  # None = fully grown
KNN_K = [100, 75, 50, 30, 20, 15, 10, 7, 5, 3, 1]
MAX_EPOCHS = 100


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
    table = pd.DataFrame(rows)
    table["value"] = pd.Series(list(values), dtype=object)  # keeps 2 as 2 and None as None
    return table


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


def plot_panel(ax, table, title, xlabel, categorical=True, tuned_value=None):
    x = [str(v) for v in table["value"]] if categorical else list(table["value"])
    style = "o-" if categorical else "-"
    ax.plot(x, table["train_RMSE"], style, color="#2a78d6", label="Training rows")
    ax.plot(x, table["val_RMSE"], style, color="#eda100", label="Validation rows")
    best = int(table["val_RMSE"].idxmin())
    ax.axvline(x[best], color="#2ca02c", linestyle=":",
               label=f"Lowest validation error: {table['value'][best]} ({table['val_RMSE'][best]:,.0f} MAD)")
    if tuned_value is not None:
        if categorical and str(tuned_value) not in x:
            print(f"  (tuned value {tuned_value} is not in the swept values {x}: add it to the list to see it marked)")
        else:
            ax.axvline(str(tuned_value) if categorical else tuned_value, color="grey", linestyle="--", label=f"Tuned value: {tuned_value}")
    ax.set(title=title, xlabel=xlabel, ylabel="RMSE (MAD)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)


def save(stage, panels, suptitle):
    """panels: list of (key, table, title, xlabel, categorical, tuned_value)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (_key, table, title, xlabel, categorical, tuned_value) in zip(axes.ravel(), panels):
        plot_panel(ax, table, title, xlabel, categorical, tuned_value)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR = RESULTS_ROOT / stage
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "complexity_curve.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tables = {key: table for key, table, *_ in panels}
    pd.concat(tables, names=["curve"]).reset_index(level=0).to_csv(OUT_DIR / "complexity_curve.csv", index=False)
    print(f"\nSaved complexity_curve.png and complexity_curve.csv in {OUT_DIR}")
