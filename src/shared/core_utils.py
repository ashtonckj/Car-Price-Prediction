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
MINIMAL_CSV = ROOT_DIR / "data" / "processed" / "cars_minimal.csv"
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


def load_minimal_data(csv_path: Path = MINIMAL_CSV):
    """
    Returns X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price.

    Loads the raw dataset with minimal preprocessing:
    - Retains raw target ('Price') without log transformation or scaling.
    - Only encodes categorical variables to numeric format using OneHotEncoder.
    - Preserves raw numerical feature scales without StandardScaler.
    """
    if not csv_path.exists():
        csv_path = PROCESSED_CSV

    df = pd.read_csv(csv_path)

    # 1. Target Extraction (Keep Raw Price)
    if "Price" in df.columns:
        y = df[["Price"]].astype("float32")
        X = df.drop(columns=["Price"])
    elif "Log Price" in df.columns:
        # If dataset only has Log Price, invert back to raw price
        y = np.expm1(df[["Log Price"]].astype("float32"))
        y.columns = ["Price"]
        X = df.drop(columns=["Log Price"])
    else:
        raise KeyError("Neither 'Price' nor 'Log Price' column found in dataset.")

    # 2. Train / Val / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=RANDOM_STATE)

    # 3. Categorical Encoding ONLY (No Feature Scaling)
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat_cols)
        ],
        remainder="passthrough"  # Leaves numeric features completely raw and unscaled
    )

    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val = preprocessor.transform(X_val).astype("float32")
    X_test = preprocessor.transform(X_test).astype("float32")

    # Convert y splits to 1D arrays
    y_train, y_val, y_test = y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

    # 4. Direct Identity Inverse Transform (Target is already in raw price)
    def inverse_transform_price(y_pred):
        return np.asarray(y_pred).ravel()

    print(f"[Minimal Data] Features: {X_train.shape[1]} | train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price


def evaluate(model_name: str, y_true, y_pred, inverse_transform_price, is_log_target: bool = False):
    """
    Evaluates regression model predictions against true ground truth values.
    
    Parameters:
    - model_name: Name of the model for print formatting.
    - y_true: True target values (either raw price or log-transformed price).
    - y_pred: Model predicted target values.
    - inverse_transform_price: Function to map model output back to original price scale (MAD).
    - is_log_target: Set to True if model was trained on log(Price), False if trained on raw Price.
    """
    # Flatten inputs to 1D arrays to guarantee shape alignment
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    
    # Transform predictions and ground truth back to original price scale (MAD)
    true_price = np.asarray(inverse_transform_price(y_true_arr)).ravel()
    pred_price = np.asarray(inverse_transform_price(y_pred_arr)).ravel()
    
    print(f"\n--- {model_name} Test Results (Original Price Scale) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(true_price, pred_price)):,.2f}")
    print(f"MAE:  {mean_absolute_error(true_price, pred_price):,.2f}")
    print(f"R2:   {r2_score(true_price, pred_price):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(true_price, pred_price) * 100:.2f}%")
    
    # Only report log-space R2 if the target was actually log-transformed during training
    if is_log_target:
        print(f"R2 (model/log space): {r2_score(y_true_arr, y_pred_arr):.4f}")
