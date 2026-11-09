import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import evaluate, load_data


class KernelRegressor:
    """Nadaraya-Watson kernel regression with a Gaussian kernel."""

    def __init__(self, bandwidth=1.0, n_neighbors=50):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        self.y_train_ = np.asarray(y, dtype=float).ravel()
        return self

    def predict(self, X):
        dist, idx = self.nn_.kneighbors(X)  # distances/indices of candidate cars
        # Subtract each row's smallest squared distance before exp(): the weighted
        # average is unchanged, but weights no longer underflow to 0 (stability).
        d2 = dist**2
        d2 = d2 - d2.min(axis=1, keepdims=True)
        weights = np.exp(-d2 / (2 * self.bandwidth**2))
        return (weights * self.y_train_[idx]).sum(axis=1) / weights.sum(axis=1)


X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = KernelRegressor()  # default: bandwidth=1.0, n_neighbors=50

t0 = time.time()
model.fit(X_train, y_train)
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("Kernel Regression", y_test, y_pred, inverse_transform_price)
