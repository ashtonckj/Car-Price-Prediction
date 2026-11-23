"""Tuned linear models (Part B): trains Ridge, Huber, and Polynomial Regression with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import full
from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import (
    choose_models,
    cross_validate,
    get_best_params,
    save_results,
    timed,
)
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    "b1_tuned": {
        "Ridge": {'alpha': 0.5623413251903491, 'fit_intercept': True, 'solver': 'sparse_cg', 'tol': 0.001},
        "Huber": {'epsilon': 2.5, 'alpha': 1e-05, 'max_iter': 100},
        "Polynomial Regression": {'degree': 2, 'interaction_only': True, 'fit_intercept': False},
    },
}


def n_numeric_columns():
    """Number of numeric columns: after encoding they are always the last columns of X."""
    return pd.read_csv(full.OUT_PATH).drop(columns="Log Price").select_dtypes(include="number").shape[1]


def build_model(name, params, n_numeric):
    if name == "Ridge":
        return Ridge(**params, random_state=RANDOM_STATE)
    if name == "Huber":
        return HuberRegressor(**params)
    if name == "Polynomial Regression":
        add_poly_terms = ColumnTransformer(
            [("numeric", PolynomialFeatures(degree=params["degree"], interaction_only=params["interaction_only"], include_bias=False),
              lambda X: list(range(X.shape[1] - n_numeric, X.shape[1])))],
            remainder="passthrough",
        )
        return Pipeline([("poly", add_poly_terms), ("linreg", LinearRegression(fit_intercept=params["fit_intercept"]))])
    raise ValueError(f"Unknown model: {name}")


def fit_predict(split, name, params, n_numeric):
    model = build_model(name, params, n_numeric)
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    y_pred = split.smear(model.predict(split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    names = choose_models(BEST_PARAMS[stage], "run")
    chosen = get_best_params(BEST_PARAMS, stage, names, "linear")
    split = load_data(stage)
    n_numeric = n_numeric_columns()

    results, cv_results = {}, {}
    for name, params in chosen.items():
        print(f"\n===== {name} (tuned) | parameters: {params} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(fold, name, params, n_numeric))
        results[name] = fit_predict(split, name, params, n_numeric)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "linear", "Linear models", results, split, cv_results, order=BEST_PARAMS[stage])


if __name__ == "__main__":
    main()
