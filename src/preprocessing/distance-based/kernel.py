import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

# ----------------------------- Settings -----------------------------
DATA_PATH = "data/processed/cars.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2     # 80% train / 20% test
CATEGORICAL_COLS = ["Brand", "Model"]

# ------------------------- Kernel Regression -------------------------
class KernelRegressor(BaseEstimator, RegressorMixin):
    """Nadaraya-Watson kernel regression with a Gaussian kernel."""

    def __init__(self, bandwidth=1.0, n_neighbors=50):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        self.y_train_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        dist, idx = self.nn_.kneighbors(X)   # distances/indices of candidate cars
        # Subtract each row's smallest squared distance before exp(): this does
        # not change the final weighted average, but stops all weights
        # underflowing to 0 for cars far from everything (numerical stability).
        d2 = dist ** 2
        d2 = d2 - d2.min(axis=1, keepdims=True)
        weights = np.exp(-d2 / (2 * self.bandwidth ** 2))
        return (weights * self.y_train_[idx]).sum(axis=1) / weights.sum(axis=1)

# ------------------------- Load and split data -----------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Price"])
y = np.log1p(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

encoder = ColumnTransformer(
    [("target_enc", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE),
      CATEGORICAL_COLS)],
    remainder="passthrough",
)

pipeline = Pipeline([
    ("encode", encoder),
    ("scale", StandardScaler()),
    ("model", KernelRegressor()),   # default: bandwidth=1.0, n_neighbors=50
])

# ----------------------------- Train ---------------------------------
t0 = time.time()
pipeline.fit(X_train, y_train)
fit_time = time.time() - t0

# ---------------------------- Predict --------------------------------
t0 = time.time()
y_pred_log = pipeline.predict(X_test)
predict_time = time.time() - t0

# ---------------------------- Evaluate -------------------------------
y_true = np.expm1(y_test)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)   # prices cannot be negative

print("\n===== Kernel Regression (default parameters) =====")
print(f"R2 (original price)  : {r2_score(y_true, y_pred):.4f}")
print(f"RMSE                 : {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
print(f"MAE                  : {mean_absolute_error(y_true, y_pred):,.2f}")
print(f"MAPE (%)             : {100 * mean_absolute_percentage_error(y_true, y_pred):.2f}")
print(f"R2 (log scale)       : {r2_score(y_test, y_pred_log):.4f}")
print(f"Fit time (s)         : {fit_time:.2f}")
print(f"Predict time (s)     : {predict_time:.2f}")