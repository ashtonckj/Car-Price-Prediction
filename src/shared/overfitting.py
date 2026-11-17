import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import PredefinedSplit, learning_curve
from sklearn.neighbors import KNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import src.models.linear_models as lm
from src.models.linear_models import polynomial_regression
from src.models.neural_models import predict as predict_torch, run_mlp
from src.preprocessing import full
from src.shared.core_utils import load_data_full


def plot_universal_learning_curve(
    model,
    split,
    model_name: str,
    train_sizes=np.linspace(0.1, 1.0, 8),
    in_mad: bool = False,
    ax=None,
):
    """
    Plots Training vs. Validation RMSE across increasing training set sizes
    using the Split object returned by load_data_full().

    Parameters
    ----------
    model : estimator
        Any scikit-learn compatible regressor (Ridge, RF, XGB, KNN, SVR, MLP).
    split : Split
        The Split namedtuple returned by load_data_full().
    model_name : str
        Title label for the plot.
    train_sizes : array-like
        Fractions (or integer counts) of split.X_train to evaluate.
    in_mad : bool, default=False
        If False, plots RMSE in Standardized Log Price (recommended for smooth curves).
        If True, converts predictions back to MAD using split.inverse_transform_price.
    ax : matplotlib.axes.Axes, optional
        Pass an axis to plot multiple models in a 2x2 grid; otherwise creates a new figure.
    """
    if callable(model) and not hasattr(model, "fit"):
        # Custom evaluation for procedural neural network models like run_mlp
        abs_train_sizes = []
        train_rmse = []
        val_rmse = []
        n_total = len(split.X_train)

        for frac in train_sizes:
            n_samples = int(frac * n_total) if frac <= 1.0 else int(frac)
            abs_train_sizes.append(n_samples)

            sub_split = split._replace(
                X_train=split.X_train[:n_samples],
                y_train=split.y_train[:n_samples],
            )
            res = model(sub_split)
            trained_net = res["model"]

            train_pred = predict_torch(trained_net, sub_split.X_train)
            val_pred = predict_torch(trained_net, split.X_val)

            if in_mad:
                tr_err = np.sqrt(mean_squared_error(
                    split.inverse_transform_price(sub_split.y_train),
                    split.inverse_transform_price(train_pred),
                ))
                v_err = np.sqrt(mean_squared_error(
                    split.inverse_transform_price(split.y_val),
                    split.inverse_transform_price(val_pred),
                ))
            else:
                tr_err = np.sqrt(mean_squared_error(sub_split.y_train.ravel(), train_pred.ravel()))
                v_err = np.sqrt(mean_squared_error(split.y_val.ravel(), val_pred.ravel()))

            train_rmse.append(tr_err)
            val_rmse.append(v_err)

        abs_train_sizes = np.array(abs_train_sizes)
        train_rmse = np.array(train_rmse)
        val_rmse = np.array(val_rmse)
        ylabel = "RMSE (MAD)" if in_mad else "RMSE (Standardized Log Price)"

    else:
        # Standard scikit-learn estimator
        # 1. Combine train and val only (keep split.X_test completely untouched)
        X_tv = np.vstack([split.X_train, split.X_val])
        y_tv = np.concatenate([split.y_train.ravel(), split.y_val.ravel()])

        # -1 = always in training pool, 0 = always in validation set
        test_fold = np.concatenate([
            np.full(len(split.X_train), -1, dtype=int),
            np.zeros(len(split.X_val), dtype=int),
        ])
        ps = PredefinedSplit(test_fold)

        # 2. Choose error scale
        if in_mad:
            def _mad_rmse(y_true, y_pred):
                true_mad = split.inverse_transform_price(y_true)
                pred_mad = split.inverse_transform_price(y_pred)
                return np.sqrt(mean_squared_error(true_mad, pred_mad))

            scoring = make_scorer(_mad_rmse, greater_is_better=False)
            ylabel = "RMSE (MAD)"
        else:
            scoring = "neg_mean_squared_error"
            ylabel = "RMSE (Standardized Log Price)"

        # 3. Compute learning curve
        abs_train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X_tv,
            y=y_tv,
            train_sizes=train_sizes,
            cv=ps,
            scoring=scoring,
            n_jobs=-1,
        )

        if in_mad:
            train_rmse = -train_scores.ravel()
            val_rmse = -val_scores.ravel()
        else:
            train_rmse = np.sqrt(-train_scores.ravel())
            val_rmse = np.sqrt(-val_scores.ravel())

    # 4. Plot
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(abs_train_sizes, train_rmse, "o-", color="#1f77b4", label="Training Error (RMSE)")
    ax.plot(abs_train_sizes, val_rmse, "o-", color="#ff7f0e", label="Validation Error (RMSE)")
    ax.set_title(f"Learning Curve: {model_name}")
    ax.set_xlabel("Training Set Size (Listings)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    if standalone:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    split = load_data_full()

    # Set numeric count for polynomial regression features
    lm.N_NUMERIC = pd.read_csv(full.OUT_PATH).drop(columns="Log Price").select_dtypes(include="number").shape[1]

    models = [
        ("Linear Family (Polynomial Regression)", polynomial_regression()),
        ("Tree Family (Random Forest)", RandomForestRegressor(max_features=0.70,n_estimators=500, max_depth=50, min_samples_leaf=5, min_samples_split=5, random_state=42, n_jobs=-1)),
        ("Distance Family (KNN)", KNeighborsRegressor(n_neighbors=12, n_jobs=-1)),
        ("Neural Network Family (MLP)", run_mlp),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for ax, (name, model) in zip(axes.ravel(), models):
        plot_universal_learning_curve(model, split, model_name=name, in_mad=False, ax=ax)

    plt.tight_layout()
    plt.show()
    plt.savefig("learning_curves1.png", dpi=300, bbox_inches="tight")