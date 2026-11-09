import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import evaluate, load_data

BATCH_SIZE = 100  # test cars solved at once (only affects speed/memory)


class LocallyWeightedRegressor:
    def __init__(self, n_neighbors=100, bandwidth=1.0, ridge=1.0):
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.ridge = ridge

    def fit(self, X, y):
        # "Training" only stores the data and builds a fast neighbour search.
        # The real work (the local regressions) happens at prediction time.
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float).ravel()
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
            xb = X[start : start + BATCH_SIZE]  # (B, d) cars to predict
            dist, idx = self.nn_.kneighbors(xb)  # (B, k) neighbours

            # Kernel weights (smallest squared distance subtracted for stability)
            d2 = dist**2
            d2 = d2 - d2.min(axis=1, keepdims=True)
            w = np.exp(-d2 / (2 * self.bandwidth**2))  # (B, k)

            # Centre neighbours on the query car and add a column of 1s (intercept)
            Z = self.X_train_[idx] - xb[:, None, :]  # (B, k, d)
            A = np.concatenate([np.ones(Z.shape[:2] + (1,)), Z], axis=2)  # (B, k, d+1)
            yn = self.y_train_[idx]  # (B, k)

            # Weighted ridge normal equations, solved for the whole batch at once
            Aw = A * w[:, :, None]
            lhs = np.einsum("bki,bkj->bij", Aw, A) + penalty  # A' W A + penalty
            rhs = np.einsum("bki,bk->bi", Aw, yn)  # A' W y
            beta = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]

            predictions[start : start + BATCH_SIZE] = beta[:, 0]  # intercept = prediction

        # A local plane can extrapolate wildly for unusual cars, so keep
        # predictions inside the range of targets seen in training.
        return np.clip(predictions, self.y_min_, self.y_max_)


X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = LocallyWeightedRegressor()  # default: k=100, bandwidth=1.0, ridge=1.0

t0 = time.time()
model.fit(X_train, y_train)
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("Locally Weighted Regression", y_test, y_pred, inverse_transform_price)
