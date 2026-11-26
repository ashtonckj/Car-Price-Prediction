"""Tuned tree-based models (Part B): trains Decision Tree, Random Forest, and Gradient Boosting with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from functools import partial
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE
from src.shared.experiment import choose_models, fit_predict, run_family
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    "b1_tuned": {
        "Decision Tree": {'max_depth': None, 'min_samples_split': 40, 'min_samples_leaf': 2, 'max_features': None, 'criterion': 'squared_error', 'ccp_alpha': 0.0},
        "Random Forest": {"n_estimators": 500, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0, "max_depth": 50, "bootstrap": True},
        "Gradient Boosting": {'learning_rate': 0.1, 'max_depth': 8, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500, 'subsample': 0.7},
    },
}


def build_model(name, params):
    if name == "Decision Tree":
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def main():
    stage = choose_stage("B")
    params = BEST_PARAMS[stage]
    names = choose_models(list(params), "run")
    models = {name: partial(fit_predict, partial(build_model, name, params[name])) for name in names}
    run_family(stage, load_data(stage), "tree", "Tree-based models", models, tuned_params=params)


if __name__ == "__main__":
    main()
