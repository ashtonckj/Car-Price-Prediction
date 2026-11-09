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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "cars.csv"
RANDOM_STATE = 42
TARGET = "Log Price"
ONE_HOT_PREFIXES = ("Brand_", "Model_")


def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    features = df.drop(columns=TARGET)
    log_price = df[TARGET]

    # 80% train+val / 20% test, then 10% of total for validation
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, log_price, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=RANDOM_STATE)

    # Scale everything except the one-hot columns (already 0/1)
    numeric_columns = [c for c in features.columns if not c.startswith(ONE_HOT_PREFIXES)]
    feature_preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), numeric_columns)],
        remainder="passthrough",
    )
    X_train = feature_preprocessor.fit_transform(X_train).astype("float32")  # fit on train only
    X_val = feature_preprocessor.transform(X_val).astype("float32")
    X_test = feature_preprocessor.transform(X_test).astype("float32")

    price_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
    scale_price = lambda y: price_scaler.transform(y.to_numpy().reshape(-1, 1)).astype("float32")
    y_train, y_val, y_test = scale_price(y_train), scale_price(y_val), scale_price(y_test)

    def inverse_transform_price(y_scaled):  # scaled target -> original price (MAD)
        y = price_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
        return np.expm1(y)

    print(f"Features: {X_train.shape[1]} | train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price


def evaluate(model_name, y_true_scaled, y_pred_scaled, inverse_transform_price):
    y_true, y_pred = inverse_transform_price(y_true_scaled), inverse_transform_price(y_pred_scaled)
    print(f"\n--- {model_name} test results (original price scale) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")
    print(f"R2 (scaled/log space): {r2_score(y_true_scaled, y_pred_scaled):.4f}")


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