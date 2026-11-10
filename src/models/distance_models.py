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

from src.shared.core_utils import evaluate, load_data
from src.shared.experiment import save_and_plot, timed

BATCH_SIZE = 100  # LWR: test cars solved at once (only affects speed/memory)

class RadiusRegressor(RadiusNeighborsRegressor):
    """Default RadiusNeighborsRegressor (radius=1.0), but cars with no neighbour
    inside the radius get the average training target instead of NaN."""
 
    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return super().fit(X, y)
 
    def predict(self, X):
        y_pred = super().predict(X)
        empty = np.isnan(y_pred)
        y_pred[empty] = self.fallback_
        print(f"Empty neighbourhoods: {100 * empty.mean():.2f}% of test cars (filled with train mean)")
        return y_pred


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


# All models use default parameters
MODELS = {
    "KNN": lambda: KNeighborsRegressor(),  # n_neighbors=5, uniform weights
    "Radius Neighbours": lambda: RadiusRegressor(),
    "Kernel Regression": lambda: KernelRegressor(),  # bandwidth=1.0, n_neighbors=50
    "Locally Weighted Regression": lambda: LocallyWeightedRegressor(),  # k=100, bandwidth=1.0, ridge=1.0
}


def main():
    X_train, y_train, _X_val, _y_val, X_test, y_test, inverse_transform_price = load_data()

    results = {}
    for name, make_model in MODELS.items():
        print(f"\n===== {name} =====")
        model = make_model()
        _, fit_time = timed(model.fit, X_train, y_train.ravel())
        y_pred, predict_time = timed(model.predict, X_test)
        print(f"Fit time (s): {fit_time:.2f} | Predict time (s): {predict_time:.2f}")
        evaluate(name, y_test, y_pred, inverse_transform_price)
        results[name] = {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}

    save_and_plot("distance", "Distance-based models", results, y_test, inverse_transform_price)


if __name__ == "__main__":
    main()