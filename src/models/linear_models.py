import sys
from pathlib import Path

from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_data
from src.shared.experiment import save_and_plot, timed

# All models use default parameters
MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(random_state=RANDOM_STATE),
    "Lasso": lambda: Lasso(random_state=RANDOM_STATE),
    "Huber": lambda: HuberRegressor(),
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