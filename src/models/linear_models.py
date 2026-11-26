"""Linear models: Ridge, Huber, Polynomial Regression. Run it and type a stage number."""
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
from src.shared.experiment import fit_predict, run_family
from src.shared.stages import choose_stage, load_data


def polynomial(degree, n_numeric):
    """Linear regression on polynomial terms of the numeric columns (the last n_numeric columns of X);
    the one-hot columns pass through unchanged. Column indices, not a lambda, so the model can be pickled."""
    add_terms = ColumnTransformer([("numeric", PolynomialFeatures(degree=degree), list(range(-n_numeric, 0)))], remainder="passthrough")
    return Pipeline([("poly", add_terms), ("linreg", LinearRegression())])


MODELS = {  # all default parameters
    "Ridge": lambda n_numeric: Ridge(random_state=RANDOM_STATE),
    "Huber": lambda n_numeric: HuberRegressor(),
    "Polynomial Regression": lambda n_numeric: polynomial(2, n_numeric),
}


def main():
    stage = choose_stage()
    split = load_data(stage)
    n_numeric = n_numeric_columns(split)
    run_family(stage, split, "linear", "Linear models", {name: partial(fit_predict, partial(make, n_numeric)) for name, make in MODELS.items()})


if __name__ == "__main__":
    main()
