import sys
from pathlib import Path
import pandas as pd

from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import BRAND_MODEL, PROCESSED_CSV, RANDOM_STATE, TARGET, evaluate, load_data
from src.shared.experiment import save_and_plot, timed

_columns = pd.read_csv(PROCESSED_CSV, nrows=0).columns
N_NUMERIC = len([c for c in _columns if c not in BRAND_MODEL + ["Fuel", TARGET]])

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
    "Polynomial Regression": polynomial_regression,
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

    save_and_plot("linear", "Linear models", results, y_test, inverse_transform_price)

if __name__ == "__main__":
    main()
