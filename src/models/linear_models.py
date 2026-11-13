"""Linear models: Ridge, Huber, Polynomial Regression. Run it and type a stage number."""
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

from src.preprocessing import full, minimal
from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data

N_NUMERIC = 0  # set in main(): number of numeric columns (always the last ones after encoding)


def polynomial_regression():
    add_poly_terms = ColumnTransformer(
        [("numeric", PolynomialFeatures(),
          lambda X: list(range(X.shape[1] - N_NUMERIC, X.shape[1])))],
        remainder="passthrough",
    )
    return Pipeline([("poly", add_poly_terms), ("linreg", LinearRegression())])


# All models use default parameters
MODELS = {
    "Ridge": lambda: Ridge(random_state=RANDOM_STATE),
    "Huber": lambda: HuberRegressor(),
    "Polynomial Regression": polynomial_regression,  # no (): the factory is called later, once per fit
}


def fit_predict(make_model, split):
    model = make_model()
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    global N_NUMERIC
    stage = choose_stage()
    split = load_data(stage)  # also creates the stage's CSV if it is missing

    csv_path, target = {"a1_baseline": (minimal.OUT_PATH, "Price"),
                        "a2_preprocessed": (full.OUT_PATH, "Log Price")}[stage]
    N_NUMERIC = pd.read_csv(csv_path, nrows=100).drop(columns=target).select_dtypes(include="number").shape[1]

    results, cv_results = {}, {}
    for name, make_model in MODELS.items():
        print(f"\n===== {name} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(make_model, fold))
        results[name] = fit_predict(make_model, split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "linear", "Linear models", results, split, cv_results)


if __name__ == "__main__":
    main()
