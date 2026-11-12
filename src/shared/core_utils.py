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

DataCV = namedtuple('DataCV', ['folds', 'X_train', 'y_train', 'X_test', 'y_test', 'inverse_transform_price', 'n_folds'])
FoldData = namedtuple('FoldData', ['X_train', 'y_train', 'X_val', 'y_val'])

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
    """Returns DataCV with 8-fold cross-validation and a fixed 20% held-out test set.

    Numeric features and the target are standardized.
    Global preprocessor (fit on all trainval) is used for test set & inverse_transform_price.
    Per-fold preprocessors are fit on each fold's training portion only.
    y arrays have shape (n, 1); use y.ravel() for sklearn models.
    """
    N_FOLDS = 5
    df = pd.read_csv(PROCESSED_CSV)
    X, y = df.drop(columns=TARGET), df[[TARGET]]

    # Hold out 20% as fixed test set (same split as before)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric = [c for c in X.columns if c not in BRAND_MODEL + ["Fuel"]]

    # --- GLOBAL preprocessor: fit on ALL trainval, used for test set ---
    X_tv_global = X_trainval.copy()
    X_test_global = X_test.copy()

    fill_from_train("Fiscal Power (CV)", [BRAND_MODEL + ["Fuel"], BRAND_MODEL], None, X_tv_global, X_test_global)
    fill_from_train("Number of Doors", [BRAND_MODEL], 5, X_tv_global, X_test_global)

    global_preprocessor = ColumnTransformer([
        ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), BRAND_MODEL),
        ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel"]),
        ("numeric", StandardScaler(), numeric),
    ])
    global_preprocessor.fit(X_tv_global)
    X_train_arr = global_preprocessor.transform(X_tv_global).astype("float32")
    X_test_arr = global_preprocessor.transform(X_test_global).astype("float32")

    global_target_scaler = StandardScaler().fit(y_trainval)
    y_train_arr = global_target_scaler.transform(y_trainval).astype("float32")
    y_test_arr = global_target_scaler.transform(y_test).astype("float32")

    def inverse_transform_price(y_scaled):  # model output -> price in MAD
        return np.expm1(global_target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))).ravel()

    # --- PER-FOLD preprocessing ---
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for train_idx, val_idx in kf.split(X_trainval):
        X_fold_train = X_trainval.iloc[train_idx].copy()
        X_fold_val = X_trainval.iloc[val_idx].copy()
        y_fold_train = y_trainval.iloc[train_idx].copy()
        y_fold_val = y_trainval.iloc[val_idx].copy()

        fill_from_train("Fiscal Power (CV)", [BRAND_MODEL + ["Fuel"], BRAND_MODEL], None, X_fold_train, X_fold_val)
        fill_from_train("Number of Doors", [BRAND_MODEL], 5, X_fold_train, X_fold_val)

        fold_preprocessor = ColumnTransformer([
            ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), BRAND_MODEL),
            ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel"]),
            ("numeric", StandardScaler(), numeric),
        ])
        X_fold_train_arr = fold_preprocessor.fit_transform(X_fold_train).astype("float32")
        X_fold_val_arr = fold_preprocessor.transform(X_fold_val).astype("float32")

        fold_target_scaler = StandardScaler().fit(y_fold_train)
        y_fold_train_arr = fold_target_scaler.transform(y_fold_train).astype("float32")
        y_fold_val_arr = fold_target_scaler.transform(y_fold_val).astype("float32")

        folds.append(FoldData(X_train=X_fold_train_arr, y_train=y_fold_train_arr,
                              X_val=X_fold_val_arr, y_val=y_fold_val_arr))

    print(f"Features: {folds[0].X_train.shape[1]} | {N_FOLDS} folds, test {len(X_test_arr)}")
    return DataCV(folds=folds, X_train=X_train_arr, y_train=y_train_arr, X_test=X_test_arr, 
                  y_test=y_test_arr, inverse_transform_price=inverse_transform_price, n_folds=N_FOLDS)


def load_minimal_data(csv_path: Path = MINIMAL_CSV):
    """Returns DataCV with 8-fold cross-validation and a fixed 20% held-out test set.

    Loads the raw dataset with minimal preprocessing:
    - Retains raw target ('Price') without log transformation or scaling.
    - Only encodes categorical variables to numeric format using OneHotEncoder.
    - Preserves raw numerical feature scales without StandardScaler.
    """
    N_FOLDS = 5
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

    # 2. Hold out 20% as fixed test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # --- GLOBAL preprocessor: fit on ALL trainval, used for test set ---
    global_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )
    global_preprocessor.fit(X_trainval)
    X_train_arr = global_preprocessor.transform(X_trainval).astype("float32")
    X_test_arr = global_preprocessor.transform(X_test).astype("float32")
    y_train_arr = y_trainval.values.ravel()
    y_test_arr = y_test.values.ravel()

    # Identity inverse transform (target is already in raw price)
    def inverse_transform_price(y_pred):
        return np.asarray(y_pred).ravel()

    # --- PER-FOLD preprocessing ---
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for train_idx, val_idx in kf.split(X_trainval):
        X_fold_train = X_trainval.iloc[train_idx].copy()
        X_fold_val = X_trainval.iloc[val_idx].copy()
        y_fold_train = y_trainval.iloc[train_idx].copy()
        y_fold_val = y_trainval.iloc[val_idx].copy()

        fold_preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat_cols)
            ],
            remainder="passthrough"
        )
        X_fold_train_arr = fold_preprocessor.fit_transform(X_fold_train).astype("float32")
        X_fold_val_arr = fold_preprocessor.transform(X_fold_val).astype("float32")

        y_fold_train_arr = y_fold_train.values.ravel()
        y_fold_val_arr = y_fold_val.values.ravel()

        folds.append(FoldData(X_train=X_fold_train_arr, y_train=y_fold_train_arr,
                              X_val=X_fold_val_arr, y_val=y_fold_val_arr))

    print(f"[Minimal Data] Features: {folds[0].X_train.shape[1]} | {N_FOLDS} folds, test {len(X_test_arr)}")
    return DataCV(folds=folds, X_train=X_train_arr, y_train=y_train_arr, X_test=X_test_arr, y_test=y_test_arr,
                  inverse_transform_price=inverse_transform_price, n_folds=N_FOLDS)


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
