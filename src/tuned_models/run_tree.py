"""Tuned tree-based models (Part B): trains Decision Tree, Random Forest, and Gradient Boosting with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import (
    choose_models,
    cross_validate,
    get_best_params,
    save_results,
    timed,
)
from src.shared.saved_models import save_model
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    "b1_tuned": {
        "Decision Tree": {'max_depth': None, 'min_samples_split': 40, 'min_samples_leaf': 2, 'max_features': None, 'criterion': 'squared_error', 'ccp_alpha': 0.0},
        "Random Forest": {"n_estimators": 500, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": 1.0, "max_depth": 50, "bootstrap": True},
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


def fit_predict(split, name, params):
    model = build_model(name, params)
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    y_pred = split.smear(model.predict(split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    names = choose_models(list(BEST_PARAMS[stage]), "run")
    chosen = get_best_params(BEST_PARAMS, stage, names, "tree")
    split = load_data(stage)

    results, cv_results = {}, {}
    for name, params in chosen.items():
        print(f"\n===== {name} (tuned) | parameters: {params} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(fold, name, params))
        results[name] = fit_predict(split, name, params)
        save_model(stage, "tree", name, results[name]["model"], split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "tree", "Tree-based models", results, split, cv_results, order=BEST_PARAMS[stage])


if __name__ == "__main__":
    main()
