"""Tuned distance-based models (Part B): trains KNN, Radius Neighbours, and Kernel Regression with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import (
    KNeighborsRegressor,
    NearestNeighbors,
    RadiusNeighborsRegressor,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import (
    choose_models,
    cross_validate,
    get_best_params,
    save_results,
    timed,
)
from src.shared.stages import choose_stage, load_data

# Copy the best row of each model's tuning CSV here after running tune_distance.py.
BEST_PARAMS = {
    "b1_tuned": {
        "KNN": {'n_neighbors': 9, 'weights': 'inv_sq', 'metric': 'manhattan'},
        "Radius Neighbours": {'radius_factor': 3.0, 'weights': 'inv_sq', 'metric': 'manhattan'},
        "Kernel Regression": {'kernel': 'gaussian', 'bandwidth_k': 20, 'bandwidth_factor': 0.3, 'metric': 'manhattan'},
    },
}

# Distance measures: label -> keyword arguments (KNeighborsRegressor / NearestNeighbors / pairwise_distances)
METRICS = {
    "manhattan": {"metric": "minkowski", "p": 1},
    "minkowski_1.5": {"metric": "minkowski", "p": 1.5},
    "euclidean": {"metric": "minkowski", "p": 2},
    "minkowski_3": {"metric": "minkowski", "p": 3},
    "cosine": {"metric": "cosine"},
}


def inverse_square(distances):
    """Neighbour weight = 1 / distance^2 (closer neighbours count much more than with plain 1 / distance).
    Handles both a 2-D array (KNN) and an object array of per-query arrays (Radius Neighbours)."""
    if distances.dtype == object:
        out = np.empty(len(distances), dtype=object)
        for i, d in enumerate(distances):
            out[i] = 1.0 / (np.square(d) + 1e-12)
        return out
    return 1.0 / (np.square(distances) + 1e-12)


WEIGHTS = {"uniform": "uniform", "distance": "distance", "inv_sq": inverse_square}


def _fast_metric(label):
    """Same distance as METRICS, but using scikit-learn's optimised routines for Manhattan / Euclidean."""
    if label in ("manhattan", "euclidean"):
        return {"metric": label}
    return METRICS[label]


def neighbour_scale(X, label, k=5, sample=2000):
    """Typical spacing between training points: median distance to the k-th nearest neighbour.
    Radius and bandwidth are expressed as multiples of this, so the grids work whatever the feature scale."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1, **_fast_metric(label)).fit(X)
    dist, _ = nn.kneighbors(X[idx])
    kth = dist[:, k]
    positive = kth[kth > 0]  # duplicated listings give distance 0
    return float(np.median(positive)) if len(positive) else 1.0


class RadiusRegressor:
    """Radius Neighbours: average all training cars within `radius_factor` x typical spacing of the query.
    Queries with no neighbour inside the radius fall back to their single nearest neighbour (sklearn would return NaN)."""

    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        X = np.asarray(X)
        kw = _fast_metric(self.params["metric"])
        self.radius_ = self.params["radius_factor"] * neighbour_scale(X, self.params["metric"])
        self.model_ = RadiusNeighborsRegressor(radius=self.radius_, weights=WEIGHTS[self.params["weights"]],
                                               algorithm="auto", n_jobs=-1, **kw).fit(X, y)
        self.fallback_ = KNeighborsRegressor(n_neighbors=1, n_jobs=-1, **kw).fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "no neighbours within radius" warning
            pred = np.asarray(self.model_.predict(X), dtype=float).ravel()
        missing = np.isnan(pred)
        if missing.any():
            pred[missing] = self.fallback_.predict(X[missing]).ravel()
        return pred


def kernel_weights(D, h, kernel):
    """Kernel weights for a distance matrix D (rows = queries). Gaussian / Laplacian are shifted by each row's
    smallest distance (it cancels in the weighted average) so they cannot underflow to all zeros."""
    if kernel == "gaussian":
        d0 = D.min(axis=1, keepdims=True)
        return np.exp(-0.5 * (np.square(D) - np.square(d0)) / h ** 2)
    if kernel == "laplacian":
        d0 = D.min(axis=1, keepdims=True)
        return np.exp(-(D - d0) / h)
    u = D / h
    if kernel == "epanechnikov":
        return np.clip(1.0 - u ** 2, 0.0, None)
    if kernel == "triangular":
        return np.clip(1.0 - u, 0.0, None)
    if kernel == "tricube":
        return np.clip(1.0 - u ** 3, 0.0, None) ** 3
    raise ValueError(f"Unknown kernel: {kernel}")


class KernelRegressor:
    """Nadaraya-Watson kernel regression: prediction = kernel-weighted average of ALL training prices.
    bandwidth_k = 0 : fixed bandwidth = bandwidth_factor x typical spacing between training points.
    bandwidth_k > 0 : adaptive bandwidth = bandwidth_factor x distance from the query to its bandwidth_k-th nearest
                      training car (narrow in crowded regions, wide in sparse ones).
    Compact kernels (epanechnikov / triangular / tricube) fall back to the nearest neighbour when no point is inside the bandwidth."""

    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y, dtype=float).ravel()
        self.scale_ = neighbour_scale(self.X_, self.params["metric"])
        return self

    def predict(self, X):
        X = np.asarray(X)
        kw = _fast_metric(self.params["metric"])
        factor, k = self.params["bandwidth_factor"], int(self.params.get("bandwidth_k", 0))
        chunk = max(1, int(1e7 // len(self.X_)))  # keeps each distance matrix at roughly 80 MB
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):
            D = pairwise_distances(X[s:s + chunk], self.X_, n_jobs=-1, **kw)
            if k > 0:
                kth = np.partition(D, k - 1, axis=1)[:, k - 1:k]
                h = np.where(kth > 0, factor * kth, factor * self.scale_)  # duplicated cars give kth = 0
            else:
                h = factor * self.scale_
            W = kernel_weights(D, h, self.params["kernel"])
            total = W.sum(axis=1)
            pred = (W @ self.y_) / np.where(total > 0, total, 1.0)
            empty = total <= 0
            if empty.any():
                pred[empty] = self.y_[D[empty].argmin(axis=1)]
            out[s:s + chunk] = pred
        return out


def build_model(name, params):
    if name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=WEIGHTS[params["weights"]],
            algorithm="auto",   # only affects speed, not the predictions
            n_jobs=-1,
            **METRICS[params["metric"]],
        )
    if name == "Radius Neighbours":
        return RadiusRegressor(params)
    if name == "Kernel Regression":
        return KernelRegressor(params)
    raise ValueError(f"Unknown model: {name}")


def predict(model, X):
    return model.predict(X).ravel()


def fit_predict(split, name, params):
    model = build_model(name, params)
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())  # only the fit; no validation predictions inside
    y_pred, predict_time = timed(predict, model, split.X_test)
    y_pred = split.smear(predict(model, split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    names = choose_models(list(BEST_PARAMS[stage]), "run")
    chosen = get_best_params(BEST_PARAMS, stage, names, "distance")
    split = load_data(stage)

    results, cv_results = {}, {}
    for name, params in chosen.items():
        print(f"\n===== {name} (tuned) | parameters: {params} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(fold, name, params))
        results[name] = fit_predict(split, name, params)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "distance", "Distance-based models", results, split, cv_results, order=BEST_PARAMS[stage])


if __name__ == "__main__":
    main()
