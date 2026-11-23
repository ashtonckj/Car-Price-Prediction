"""Tuned distance-based models (Part B): trains KNN, Radius Neighbours, and Kernel Regression with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import evaluate
from src.shared.experiment import (
    choose_models,
    cross_validate,
    get_best_params,
    save_results,
    timed,
)
from src.shared.saved_models import save_model
from src.shared.stages import choose_stage, load_data

# Copy the best row of results/<stage>/distance_tuning_knn.csv here after running tune_distance.py
BEST_PARAMS = {
    "b1_tuned": {
        "KNN": {'n_neighbors': 9, 'weights': 'inv_sq', 'metric': 'manhattan'},
    },
}

# Distance measures: label -> KNeighborsRegressor keyword arguments
METRICS = {
    "manhattan": {"metric": "minkowski", "p": 1},
    "minkowski_1.5": {"metric": "minkowski", "p": 1.5},
    "euclidean": {"metric": "minkowski", "p": 2},
    "minkowski_3": {"metric": "minkowski", "p": 3},
    "cosine": {"metric": "cosine"},
}


def inverse_square(distances):
    """Neighbour weight = 1 / distance^2 (closer neighbours count much more than with plain 1 / distance)."""
    return 1.0 / (np.square(distances) + 1e-12)


WEIGHTS = {"uniform": "uniform", "distance": "distance", "inv_sq": inverse_square}


def build_model(name, params):
    if name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=WEIGHTS[params["weights"]],
            algorithm="auto",   # only affects speed, not the predictions
            n_jobs=-1,
            **METRICS[params["metric"]],
        )
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
        save_model(stage, "distance", name, results[name]["model"], split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "distance", "Distance-based models", results, split, cv_results, order=BEST_PARAMS[stage])


if __name__ == "__main__":
    # Run main() from the module imported as src.tuned_models.run_distance, so saved models refer to its classes / functions by
    # that name and not by "__main__" (otherwise only this script could load them; see src/shared/saved_models.py)
    from src.tuned_models import run_distance
    run_distance.main()
