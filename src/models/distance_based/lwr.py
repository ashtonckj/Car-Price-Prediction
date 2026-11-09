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
BATCH_SIZE = 500    # test cars solved at once (only affects speed/memory)

# --------------------- Locally Weighted Regression -------------------
class LocallyWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=100, bandwidth=1.0, ridge=1.0):
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.ridge = ridge

    def fit(self, X, y):
        # "Training" only stores the data and builds a fast neighbour search.
        # The real work (the local regressions) happens at prediction time.
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_train_)
        self.y_min_, self.y_max_ = self.y_train_.min(), self.y_train_.max()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        predictions = np.empty(len(X))

        # Penalty matrix: ridge on all coefficients EXCEPT the intercept
        penalty = self.ridge * np.eye(n_features + 1)
        penalty[0, 0] = 0.0

        for start in range(0, len(X), BATCH_SIZE):
            xb = X[start:start + BATCH_SIZE]                 # (B, d) cars to predict
            dist, idx = self.nn_.kneighbors(xb)              # (B, k) neighbours

            # Kernel weights (subtract the smallest squared distance first for
            # numerical stability - it does not change the fitted line).
            d2 = dist ** 2
            d2 = d2 - d2.min(axis=1, keepdims=True)
            w = np.exp(-d2 / (2 * self.bandwidth ** 2))     # (B, k)

            # Centre neighbours on the query car and add a column of 1s (intercept)
            Z = self.X_train_[idx] - xb[:, None, :]          # (B, k, d)
            A = np.concatenate([np.ones(Z.shape[:2] + (1,)), Z], axis=2)  # (B, k, d+1)
            yn = self.y_train_[idx]                          # (B, k)

            # Weighted ridge normal equations, solved for the whole batch at once
            Aw = A * w[:, :, None]
            lhs = np.einsum("bki,bkj->bij", Aw, A) + penalty  # A' W A + penalty
            rhs = np.einsum("bki,bk->bi", Aw, yn)             # A' W y
            beta = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]

            predictions[start:start + BATCH_SIZE] = beta[:, 0]  # intercept = prediction

        # A local plane can extrapolate wildly for unusual cars, so keep
        # predictions inside the range of prices seen in training.
        return np.clip(predictions, self.y_min_, self.y_max_)

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
    ("model", LocallyWeightedRegressor()),   # default: k=100, bandwidth=1.0, ridge=1.0
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

print("\n===== Locally Weighted Regression (default parameters) =====")
print(f"R2 (original price)  : {r2_score(y_true, y_pred):.4f}")
print(f"RMSE                 : {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
print(f"MAE                  : {mean_absolute_error(y_true, y_pred):,.2f}")
print(f"MAPE (%)             : {100 * mean_absolute_percentage_error(y_true, y_pred):.2f}")
print(f"R2 (log scale)       : {r2_score(y_test, y_pred_log):.4f}")
print(f"Fit time (s)         : {fit_time:.2f}")
print(f"Predict time (s)     : {predict_time:.2f}")