"""Tuned KNN (Part B): plain distance-based KNeighborsRegressor trained with BEST_PARAMS; reports 5-fold CV and test results."""
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data

# Placeholder values: replace with the best row from results/tuning/knn_<stage>.csv after running tune_knn.py
BEST_PARAMS = {
    "b1_tuned": {'n_neighbors': 9, 'weights': 'inv_sq', 'metric': 'manhattan'},
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


def build_model(params):
    return KNeighborsRegressor(
        n_neighbors=params["n_neighbors"],
        weights=WEIGHTS[params["weights"]],
        algorithm="auto",   # only affects speed, not the predictions
        n_jobs=-1,
        **METRICS[params["metric"]],
    )


def predict(model, X):
    return model.predict(X).ravel()


def train_model(params, split):
    """Fit KNN (stores the training set). Returns (model, val MSE) -- same signature as the DCN-V2 version."""
    model = build_model(params)
    model.fit(split.X_train, split.y_train.ravel())
    val_loss = float(np.mean((predict(model, split.X_val) - split.y_val.ravel()) ** 2))
    return model, val_loss


def fit_predict(split, params):
    (model, _), fit_time = timed(train_model, params, split)
    y_pred, predict_time = timed(predict, model, split.X_test)
    y_pred = split.smear(predict(model, split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    if stage not in BEST_PARAMS:
        sys.exit(f"Add the best parameters for {stage} to BEST_PARAMS at the top of this file (see results/tuning/knn_{stage}.csv).")
    params = BEST_PARAMS[stage]
    print(f"Parameters: {params}")
    split = load_data(stage)

    name = "KNN"
    print(f"\n===== {name} (tuned) =====")
    cv_results = {name: cross_validate(stage, lambda fold: fit_predict(fold, params))}
    results = {name: fit_predict(split, params)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "knn", "Distance-based", results, split, cv_results)


if __name__ == "__main__":
    main()