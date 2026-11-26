"""Distance-based models: KNN, Radius Neighbours, Kernel Regression. Run it and type a stage number."""
import sys
from functools import partial
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

from src.shared.experiment import fit_predict, run_family
from src.shared.stages import choose_stage, load_data


class RadiusRegressor(RadiusNeighborsRegressor):
    """Default RadiusNeighborsRegressor (radius=1.0), but cars with no neighbour inside the radius get the average training target instead of NaN."""

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
        dist, idx = self.nn_.kneighbors(X)
        # Subtract each row's smallest squared distance before exp(): the weighted average is unchanged,
        # but the weights can no longer all underflow to 0
        d2 = dist**2
        d2 = d2 - d2.min(axis=1, keepdims=True)
        weights = np.exp(-d2 / (2 * self.bandwidth**2))
        return (weights * self.y_train_[idx]).sum(axis=1) / weights.sum(axis=1)


# All models use default parameters
MODELS = {
    "KNN": KNeighborsRegressor,  # n_neighbors=5, uniform weights
    "Radius Neighbours": RadiusRegressor,
    "Kernel Regression": KernelRegressor,  # bandwidth=1.0, n_neighbors=50
}


def main():
    stage = choose_stage()
    split = load_data(stage)
    run_family(stage, split, "distance", "Distance-based models", {name: partial(fit_predict, make) for name, make in MODELS.items()})


if __name__ == "__main__":
    main()
