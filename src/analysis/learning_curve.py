import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.models.linear_models as lm
from src.models.neural_models import predict as predict_torch
from src.preprocessing import full
from src.shared.core_utils import RANDOM_STATE, load_data_full

OUT_DIR = ROOT_DIR / "results" / "overfitting"


def plot_complexity_curve(
    model_builder,
    param_values,
    split,
    model_name: str,
    xlabel: str,
    invert_x: bool = False,
    log_x: bool = False,
    in_mad: bool = False,
    ax=None,
):
    """
    Plots Training vs. Validation RMSE across a complexity parameter range
    on a FIXED training sample (split.X_train) and validation set (split.X_val).
    """
    train_rmse, val_rmse = [], []

    for val in param_values:
        model = model_builder(val)

        if callable(model) and not hasattr(model, "fit"):
            # PyTorch procedural model (e.g., run_mlp)
            res = model(split)
            trained_net = res["model"]
            train_pred = predict_torch(trained_net, split.X_train)
            val_pred = predict_torch(trained_net, split.X_val)
        else:
            # Standard scikit-learn estimator / Pipeline
            model.fit(split.X_train, split.y_train.ravel())
            train_pred = model.predict(split.X_train)
            val_pred = model.predict(split.X_val)

        if in_mad:
            tr_err = np.sqrt(mean_squared_error(
                split.inverse_transform_price(split.y_train),
                split.inverse_transform_price(train_pred),
            ))
            v_err = np.sqrt(mean_squared_error(
                split.inverse_transform_price(split.y_val),
                split.inverse_transform_price(val_pred),
            ))
        else:
            tr_err = np.sqrt(mean_squared_error(split.y_train.ravel(), np.asarray(train_pred).ravel()))
            v_err = np.sqrt(mean_squared_error(split.y_val.ravel(), np.asarray(val_pred).ravel()))

        train_rmse.append(tr_err)
        val_rmse.append(v_err)

    standalone = ax is None
    if standalone:
        _fig, ax = plt.subplots(figsize=(8, 5))

    x_ticks = [str(v) for v in param_values] if isinstance(param_values[0], tuple) else param_values
    ax.plot(x_ticks, train_rmse, "o-", color="#1f77b4", label="Training Error (RMSE)")
    ax.plot(x_ticks, val_rmse, "o-", color="#ff7f0e", label="Validation Error (RMSE)")

    if log_x and not isinstance(param_values[0], tuple):
        ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()

    # Mark optimal complexity (lowest validation RMSE)
    best_idx = int(np.argmin(val_rmse))
    ax.axvline(x_ticks[best_idx], color="green", linestyle=":", alpha=0.8, label=f"Best Val ({param_values[best_idx]}: {val_rmse[best_idx]:.3f})")

    ax.set_title(f"Complexity Curve: {model_name}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (MAD)" if in_mad else "RMSE (Standardized Log Price)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    if standalone:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    split = load_data_full()
    lm.N_NUMERIC = pd.read_csv(full.OUT_PATH).drop(columns="Log Price").select_dtypes(include="number").shape[1]

    # 1. Polynomial Regression: Sweep polynomial degree (degree 1 = simple linear, high degree = complex)
    degrees = [1, 2, 3, 4]

    def build_poly(degree):
        add_poly_terms = ColumnTransformer(
            [("numeric", PolynomialFeatures(degree=degree),
              lambda X: list(range(X.shape[1] - lm.N_NUMERIC, X.shape[1])))],
            remainder="passthrough",
        )
        return Pipeline([("poly", add_poly_terms), ("linreg", LinearRegression())])

    # 2. Tree Family: Sweep max_depth (shallow = underfit, deep = overfit)
    depths = [2, 4, 6, 8, 12, 16, 22, 30, 45]
    build_rf = lambda d: RandomForestRegressor(
        n_estimators=200, max_depth=d, min_samples_leaf=1, max_features=0.7, random_state=42, n_jobs=-1
    )

    # 3. Distance Family: Sweep n_neighbors k (large k = smooth/underfit, small k = overfit)
    k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]  # More granular sweep
    build_knn = lambda k: KNeighborsRegressor(n_neighbors=k, n_jobs=-1)

    # 4. Neural Network Family: Sweep training epochs (few epochs = underfit, many epochs = overfit)
    epochs_list = [2, 5, 10, 20, 35, 50, 75, 100]

    def build_mlp(epochs):
        def _train(s):
            torch.manual_seed(RANDOM_STATE)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = nn.Sequential(
                nn.Linear(s.X_train.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            loader = DataLoader(
                TensorDataset(torch.tensor(s.X_train), torch.tensor(s.y_train)),
                batch_size=256,
                shuffle=True,
            )
            for _ in range(epochs):
                model.train()
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    loss_fn(model(X_batch), y_batch).backward()
                    optimizer.step()
            return {"model": model}

        return _train

    configs = [
        (build_poly, degrees, "Polynomial Regression", "Polynomial Degree (Low -> High Complexity)", False, False),
        (build_rf, depths, "Tree Family (Random Forest)", "Max Tree Depth (Low -> High Complexity)", False, False),
        (build_knn, k_values, "Distance Family (KNN)", "Number of Neighbors k (Inverted: Low -> High Complexity)", True, False),
        (build_mlp, epochs_list, "Neural Network Family (MLP)", "Training Epochs (Low -> High Complexity)", False, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (builder, params, name, xlabel, inv_x, log_x) in zip(axes.ravel(), configs):
        plot_complexity_curve(
            builder, params, split,
            model_name=name, xlabel=xlabel, invert_x=inv_x, log_x=log_x, in_mad=False, ax=ax
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "complexity_curves1.png", dpi=300, bbox_inches="tight")
    plt.show()