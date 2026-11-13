"""Tree-based models: Decision Tree, Random Forest, Gradient Boosting. Run it and type a stage number."""
import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data

# All models use default parameters
MODELS = {
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),  # n_jobs only changes speed
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def fit_predict(make_model, split):
    model = make_model()
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage()
    split = load_data(stage)

    results, cv_results = {}, {}
    for name, make_model in MODELS.items():
        print(f"\n===== {name} =====")
        cv_results[name] = cross_validate(stage, lambda fold: fit_predict(make_model, fold))
        results[name] = fit_predict(make_model, split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "tree", "Tree-based models", results, split, cv_results)


if __name__ == "__main__":
    main()
