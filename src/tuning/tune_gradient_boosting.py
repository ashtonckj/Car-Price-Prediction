"""Grid search for the Gradient Boosting's hyperparameters on a Part B stage."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sklearn.ensemble import GradientBoostingRegressor

from src.shared.core_utils import RANDOM_STATE
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores

# 2 x 1 x 3 x 5 x 3 x 5 = 450 combinations
PARAM_GRID = {
    "learning_rate": [ 0.01, 0.05, 0.1],
    "max_depth": [ 3, 5, 8 ],
    "max_features": [ 1.0, "sqrt" ],
    "min_samples_leaf": [ 1, 4],
    "min_samples_split": [ 2, 10 ],
    "n_estimators": [ 100, 300, 500 ],
    "subsample": [ 0.7, 0.85, 1.0 ],
}


def main():
    stage = choose_stage("B")
    split = load_data(stage)

    def evaluate(params):
        rf = GradientBoostingRegressor(**params, random_state=42)
        rf.fit(split.X_train, split.y_train.ravel())
        y_pred = rf.predict(split.X_val)
        return validation_scores(split, y_pred)

    table, path = grid_search("gradient_boosting", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_gradient_boosting.py and run it.")


if __name__ == "__main__":
    main()
