import sys
from pathlib import Path

from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE


def _runner(make_model):
    def run(X_train, y_train, X_val, y_val, X_test):
        return make_model().fit(X_train, y_train.ravel()).predict(X_test)
    return run


# All models use default parameters
MODELS = {
    "Linear Regression": _runner(lambda: LinearRegression()),
    "Ridge": _runner(lambda: Ridge(random_state=RANDOM_STATE)),
    "Lasso": _runner(lambda: Lasso(random_state=RANDOM_STATE)),
    "Huber": _runner(lambda: HuberRegressor()),
}