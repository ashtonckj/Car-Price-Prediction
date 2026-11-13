"""Data loading (one load_data_* function per dataset) and metrics.

Every loader returns a Split and takes `fold`:
    fold=None  -> final split: 70% train / 10% val / 20% test
    fold=0..4  -> one cross-validation fold: the held-out fifth of the 70% train part becomes X_test,
                  the other four fifths are X_train, and the 10% val set is unchanged.
Imputation, encoding and scaling are always fitted on that split's training rows only.
y has shape (n, 1) (use y.ravel() for sklearn); inverse_transform_price(output) gives the price in MAD.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import full, minimal

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT_DIR / "results"
RANDOM_STATE = 42
N_FOLDS = 5
BRAND_MODEL = ["Brand", "Model"]

Split = namedtuple("Split", ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "inverse_transform_price"])


def read_processed(csv_path, create):
    """Read a processed CSV, running its preprocessing script (`create`) first if the file is missing."""
    if not csv_path.exists():
        print(f"{csv_path.name} not found, creating it...")
        create()
    return pd.read_csv(csv_path)


def split_rows(X, y, fold=None):
    """Split the rows 70/10/20 (or into a CV fold, see the module docstring). Same for every dataset."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=RANDOM_STATE)
    if fold is not None:
        fit_idx, held_out_idx = list(KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_train))[fold]
        X_train, X_test = X_train.iloc[fit_idx], X_train.iloc[held_out_idx]
        y_train, y_test = y_train.iloc[fit_idx], y_train.iloc[held_out_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test


def fill_from_train(column, key_sets, fallback, train, *others):
    """Fill NaNs in `column` with the most common value of the same group, learned from train only."""
    tables = [train.dropna(subset=[column]).groupby(keys)[column].agg(lambda s: s.mode().iloc[0]) for keys in key_sets]
    fallback = train[column].median() if fallback is None else fallback
    for df in (train, *others):
        for keys, table in zip(key_sets, tables):
            df[column] = df[column].fillna(pd.Series(table.reindex(pd.MultiIndex.from_frame(df[keys])).to_numpy(), index=df.index))
        df[column] = df[column].fillna(fallback)


def load_data_minimal(fold=None):
    """a1_baseline: one-hot encode every text column, nothing else. Raw Price target (MAD), no scaling."""
    df = read_processed(minimal.OUT_PATH, minimal.main)
    X, y = df.drop(columns="Price"), df[["Price"]]
    X_train, y_train, X_val, y_val, X_test, y_test = split_rows(X, y, fold)

    text_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [("text", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), text_cols)],
        remainder="passthrough",  # numeric columns unchanged
    )
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val, X_test = preprocessor.transform(X_val).astype("float32"), preprocessor.transform(X_test).astype("float32")
    y_train, y_val, y_test = (a.to_numpy().astype("float32") for a in (y_train, y_val, y_test))

    def inverse_transform_price(y_pred):  # target is already the price
        return np.asarray(y_pred).ravel()

    return Split(X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price)


def load_data_full(fold=None):
    """a2_preprocessed: group-mode imputation, one-hot Brand/Model/Fuel, standardised numeric features and Log Price target."""
    df = read_processed(full.OUT_PATH, full.main)
    X, y = df.drop(columns="Log Price"), df[["Log Price"]]
    X_train, y_train, X_val, y_val, X_test, y_test = split_rows(X, y, fold)
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    fill_from_train("Fiscal Power (CV)", [BRAND_MODEL + ["Fuel"], BRAND_MODEL], None, X_train, X_val, X_test)
    fill_from_train("Number of Doors", [BRAND_MODEL], 5, X_train, X_val, X_test)

    numeric = [c for c in X_train.columns if c not in BRAND_MODEL + ["Fuel"]]
    preprocessor = ColumnTransformer([
        ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), BRAND_MODEL),
        ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel"]),
        ("numeric", StandardScaler(), numeric),
    ])
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val, X_test = preprocessor.transform(X_val).astype("float32"), preprocessor.transform(X_test).astype("float32")

    target_scaler = StandardScaler().fit(y_train)
    y_train, y_val, y_test = (target_scaler.transform(a).astype("float32") for a in (y_train, y_val, y_test))

    def inverse_transform_price(y_scaled):  # scaled log price -> price in MAD
        return np.expm1(target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))).ravel()

    return Split(X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price)


def compute_metrics(y_true, y_pred, inverse_transform_price):
    """All metrics on the price in MAD, so every stage is comparable."""
    true_price = inverse_transform_price(np.asarray(y_true).ravel())
    pred_price = inverse_transform_price(np.asarray(y_pred).ravel())
    return {
        "RMSE": float(np.sqrt(mean_squared_error(true_price, pred_price))),
        "MAE": float(mean_absolute_error(true_price, pred_price)),
        "R2": float(r2_score(true_price, pred_price)),
        "MAPE": float(mean_absolute_percentage_error(true_price, pred_price) * 100),
        # R2 of log(price) from the MAD prices, so it means the same in every stage (negative predictions clipped to 0)
        "R2_log": float(r2_score(np.log1p(true_price), np.log1p(np.clip(pred_price, 0, None)))),
    }


def evaluate(model_name, y_true, y_pred, inverse_transform_price):
    """Print the test metrics and return them as a dict."""
    metrics = compute_metrics(y_true, y_pred, inverse_transform_price)
    print(f"\n--- {model_name} test results (price in MAD) ---")
    print(f"RMSE: {metrics['RMSE']:,.2f}")
    print(f"MAE:  {metrics['MAE']:,.2f}")
    print(f"R2:   {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R2 (log price): {metrics['R2_log']:.4f}")
    return metrics
