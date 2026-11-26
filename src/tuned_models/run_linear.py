"""Tuned linear models (Part B): trains Ridge, Huber, and Polynomial Regression with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from functools import partial
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, n_numeric_columns
from src.shared.experiment import choose_models, fit_predict, run_family
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    "b1_tuned": {
        "Ridge": {'alpha': 0.5623413251903491, 'fit_intercept': True, 'solver': 'sparse_cg', 'tol': 0.001},
        "Huber": {'epsilon': 2.5, 'alpha': 1e-05, 'max_iter': 100},
        "Polynomial Regression": {'degree': 2, 'interaction_only': True, 'fit_intercept': True},
    },
}


def build_model(name, params, n_numeric):
    if name == "Ridge":
        return Ridge(**params, random_state=RANDOM_STATE)
    if name == "Huber":
        return HuberRegressor(**params)
    if name == "Polynomial Regression":
        add_poly_terms = ColumnTransformer(
            [("numeric", PolynomialFeatures(degree=params["degree"], interaction_only=params["interaction_only"], include_bias=False),
              list(range(-n_numeric, 0)))],  # the numeric columns (indices, not a lambda, so the model can be pickled)
            remainder="passthrough",
        )
        return Pipeline([("poly", add_poly_terms), ("linreg", LinearRegression(fit_intercept=params["fit_intercept"]))])
    raise ValueError(f"Unknown model: {name}")


def main():
    stage = choose_stage("B")
    params = BEST_PARAMS[stage]
    names = choose_models(list(params), "run")
    split = load_data(stage)
    n_numeric = n_numeric_columns(split)
    models = {name: partial(fit_predict, partial(build_model, name, params[name], n_numeric)) for name in names}
    run_family(stage, split, "linear", "Linear models", models, tuned_params=params)


if __name__ == "__main__":
    main()
