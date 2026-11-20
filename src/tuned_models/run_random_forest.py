"""Tuned Random Forest (Part B): trains the forest with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    #"b1_tuned": {"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": 1.0, "max_depth": 15, "bootstrap": True},
    "b1_tuned": {"n_estimators": 500, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": 1.0, "max_depth": 50, "bootstrap": True},
}


def fit_predict(split, params):
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    y_pred = split.smear(model.predict(split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    if stage not in BEST_PARAMS:
        sys.exit(f"Add the best parameters for {stage} to BEST_PARAMS at the top of this file (see results/tuning/random_forest_{stage}.csv).")
    params = BEST_PARAMS[stage]
    print(f"Parameters: {params}")
    split = load_data(stage)

    name = "Random Forest"
    print(f"\n===== {name} (tuned) =====")
    cv_results = {name: cross_validate(stage, lambda fold: fit_predict(fold, params))}
    results = {name: fit_predict(split, params)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "random_forest_50depth", "Tree-based models", results, split, cv_results)


if __name__ == "__main__":
    main()
