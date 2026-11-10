import sys
from pathlib import Path

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

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
    "Decision Tree": _runner(lambda: DecisionTreeRegressor(random_state=RANDOM_STATE)),
    "Random Forest": _runner(lambda: RandomForestRegressor(random_state=RANDOM_STATE)),
    "Extra Trees": _runner(lambda: ExtraTreesRegressor(random_state=RANDOM_STATE)),
    "Gradient Boosting": _runner(lambda: GradientBoostingRegressor(random_state=RANDOM_STATE)),
}