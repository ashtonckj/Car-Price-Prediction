"""Distance-based models: KNN, Radius Neighbours, Kernel Regression. Run it and type a stage number."""
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

from src.shared.core_utils import evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data


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


# All models use default parameters
MODELS = {
    "KNN": lambda: KNeighborsRegressor(),  # n_neighbors=5, uniform weights
    "Radius Neighbours": lambda: RadiusRegressor(),
    "Kernel Regression": lambda: KernelRegressor(),  # bandwidth=1.0, n_neighbors=50
}


def fit_predict(make_model, split):
    model = make_model()
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage()
    split = load_data(stage)

    results, cv_results = {}, {}
    for name, make_model in MODELS.items():
        print(f"\n===== {name} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(make_model, fold))
        results[name] = fit_predict(make_model, split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "distance", "Distance-based models", results, split, cv_results)


if __name__ == "__main__":
    main()
