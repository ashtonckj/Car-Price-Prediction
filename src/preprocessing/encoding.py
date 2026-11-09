from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "cars.csv"
RANDOM_STATE = 42
TARGET = "Log Price"

BRAND_MODEL_COLUMNS = ["Brand", "Model"]
FUEL_COLUMN = ["Fuel"]
MIN_FREQUENCY = 10  # Brand/Model values with fewer TRAIN rows are grouped into one "infrequent" column


class GroupModeImputer:
    """Fills missing values of one column with the most common value of the same group.

    key_sets are tried in order (most specific first). Anything still missing
    gets `fallback` (or the train median if fallback is None).
    All modes are learned from the data passed to fit() (train only).
    """

    def __init__(self, column, key_sets, fallback=None):
        self.column, self.key_sets, self.fallback = column, key_sets, fallback

    def fit(self, df):
        known = df.dropna(subset=[self.column])
        self.modes_ = [known.groupby(keys)[self.column].agg(lambda s: s.mode().iloc[0]) for keys in self.key_sets]
        self.fallback_ = self.fallback if self.fallback is not None else known[self.column].median()
        return self

    def transform(self, df):
        df = df.copy()
        for keys, modes in zip(self.key_sets, self.modes_):
            lookup = modes.reindex(pd.MultiIndex.from_frame(df[keys]))
            df[self.column] = df[self.column].fillna(pd.Series(lookup.to_numpy(), index=df.index))
        df[self.column] = df[self.column].fillna(self.fallback_)
        return df


def load_data(scale=True):
    """Returns X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price.

    scale=True  -> numeric features and target are standardized (use for neural networks)
    scale=False -> numeric features and target are left as they are (fine for tree ensembles)
    y arrays have shape (n, 1); use y.ravel() for sklearn models.
    """
    df = pd.read_csv(PROCESSED_CSV)
    features = df.drop(columns=TARGET)
    log_price = df[TARGET]

    # 70% train / 10% val / 20% test
    X_train, X_test, y_train, y_test = train_test_split(features, log_price, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=RANDOM_STATE)

    # Imputation: modes are learned from train only, then applied to all three sets
    imputers = [
        GroupModeImputer("Fiscal Power (CV)", [["Brand", "Model", "Fuel"], ["Brand", "Model"]]),
        GroupModeImputer("Number of Doors", [["Brand", "Model"]], fallback=5),
    ]
    for imputer in imputers:
        imputer.fit(X_train)
        X_train, X_val, X_test = imputer.transform(X_train), imputer.transform(X_val), imputer.transform(X_test)

    # Encoding + scaling: everything is fit on train only
    categorical_columns = BRAND_MODEL_COLUMNS + FUEL_COLUMN
    numeric_columns = [c for c in features.columns if c not in categorical_columns]
    feature_preprocessor = ColumnTransformer([
        ("brand_model", OneHotEncoder(min_frequency=MIN_FREQUENCY, handle_unknown="infrequent_if_exist",
                                      sparse_output=False), BRAND_MODEL_COLUMNS),
        ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FUEL_COLUMN),
        ("numeric", StandardScaler() if scale else "passthrough", numeric_columns),
    ])
    X_train = feature_preprocessor.fit_transform(X_train).astype("float32")
    X_val = feature_preprocessor.transform(X_val).astype("float32")
    X_test = feature_preprocessor.transform(X_test).astype("float32")

    # Target (already log1p from preprocessing). Scaler is fit on train only.
    price_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1)) if scale else None

    def prepare_target(y):
        y = y.to_numpy().reshape(-1, 1)
        return (price_scaler.transform(y) if scale else y).astype("float32")

    y_train, y_val, y_test = prepare_target(y_train), prepare_target(y_val), prepare_target(y_test)

    def inverse_transform_price(y_model_output):  # model output -> original price (MAD)
        y = np.asarray(y_model_output).reshape(-1, 1)
        y = price_scaler.inverse_transform(y) if scale else y
        return np.expm1(y).ravel()

    print(f"Features: {X_train.shape[1]} | train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price


def evaluate(model_name, y_true_scaled, y_pred_scaled, inverse_transform_price):
    y_true, y_pred = inverse_transform_price(y_true_scaled), inverse_transform_price(y_pred_scaled)
    print(f"\n--- {model_name} test results (original price scale) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")
    print(f"R2 (scaled/log space): {r2_score(y_true_scaled, np.asarray(y_pred_scaled).reshape(-1, 1)):.4f}")


def train_torch(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=1e-3, batch_size=256, patience=10):
    torch.manual_seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer, loss_function = torch.optim.Adam(model.parameters(), lr=learning_rate), nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(); loss_function(model(X_batch), y_batch).backward(); optimizer.step()
        val_loss = float(np.mean((predict(model, X_val) - y_val) ** 2))  # batched, same as MSE
        print(f"epoch {epoch + 1} | val_loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improvement = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, X, batch_size=512):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[start:start + batch_size]).to(device)
            predictions.append(model(X_batch).cpu().numpy())
    return np.concatenate(predictions)