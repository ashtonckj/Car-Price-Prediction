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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "cars.csv"
RANDOM_STATE = 42
TARGET = "Log Price"
BRAND_MODEL = ["Brand", "Model"]


def fill_from_train(column, key_sets, fallback, train, *others):
    """Fill NaNs in `column` with the most common value of the same group, learned from train only."""
    tables = [train.dropna(subset=[column]).groupby(keys)[column].agg(lambda s: s.mode().iloc[0]) for keys in key_sets]
    fallback = train[column].median() if fallback is None else fallback
    for df in (train, *others):
        for keys, table in zip(key_sets, tables):
            df[column] = df[column].fillna(pd.Series(table.reindex(pd.MultiIndex.from_frame(df[keys])).to_numpy(), index=df.index))
        df[column] = df[column].fillna(fallback)


def load_data():
    """Returns X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price.

    Numeric features and the target are standardized (fit on train only).
    y has shape (n, 1); use y.ravel() for sklearn models.
    """
    df = pd.read_csv(PROCESSED_CSV)
    X, y = df.drop(columns=TARGET), df[[TARGET]]

    # 70% train / 10% val / 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=RANDOM_STATE)
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    # Everything below learns from train only
    fill_from_train("Fiscal Power (CV)", [BRAND_MODEL + ["Fuel"], BRAND_MODEL], None, X_train, X_val, X_test)
    fill_from_train("Number of Doors", [BRAND_MODEL], 5, X_train, X_val, X_test)

    numeric = [c for c in X.columns if c not in BRAND_MODEL + ["Fuel"]]
    preprocessor = ColumnTransformer([
        ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), BRAND_MODEL),
        ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel"]),
        ("numeric", StandardScaler(), numeric),
    ])
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val, X_test = preprocessor.transform(X_val).astype("float32"), preprocessor.transform(X_test).astype("float32")

    target_scaler = StandardScaler().fit(y_train)
    y_train, y_val, y_test = (target_scaler.transform(a).astype("float32") for a in (y_train, y_val, y_test))

    def inverse_transform_price(y_scaled):  # model output -> price in MAD
        return np.expm1(target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))).ravel()

    print(f"Features: {X_train.shape[1]} | train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price


def evaluate(model_name, y_true, y_pred, inverse_transform_price):
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    true_price, pred_price = inverse_transform_price(y_true), inverse_transform_price(y_pred)
    print(f"\n--- {model_name} test results (original price scale) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(true_price, pred_price)):,.2f}")
    print(f"MAE:  {mean_absolute_error(true_price, pred_price):,.2f}")
    print(f"R2:   {r2_score(true_price, pred_price):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(true_price, pred_price) * 100:.2f}%")
    print(f"R2 (model/log space): {r2_score(y_true, y_pred):.4f}")
