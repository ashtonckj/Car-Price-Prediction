import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import (
    KNeighborsRegressor,
    NearestNeighbors,
    RadiusNeighborsRegressor,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


BATCH_SIZE = 100  # LWR: test cars solved at once (only affects speed/memory)


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


def _runner(make_model):
    def run(X_train, y_train, X_val, y_val, X_test):
        return make_model().fit(X_train, y_train.ravel()).predict(X_test)
    return run


def run_radius(X_train, y_train, X_val, y_val, X_test):
    y_pred = RadiusNeighborsRegressor().fit(X_train, y_train.ravel()).predict(X_test)  # default radius=1.0
    # Cars with no neighbour inside the radius come back as NaN
    empty = np.isnan(y_pred)
    y_pred[empty] = y_train.mean()  # fallback: average training target
    print(f"Empty neighbourhoods: {100 * empty.mean():.2f}% of test cars (filled with train mean)")
    return y_pred


MODELS = {
    "KNN": _runner(lambda: KNeighborsRegressor()),  # n_neighbors=5, uniform weights
    "Radius Neighbours": run_radius,
    "Kernel Regression": _runner(lambda: KernelRegressor()),  # bandwidth=1.0, n_neighbors=50
    "Locally Weighted Regression": _runner(lambda: LocallyWeightedRegressor()),  # k=100, bandwidth=1.0, ridge=1.0
}