"""Grid search for the Random Forest's hyperparameters on a Part B stage."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sklearn.ensemble import RandomForestRegressor

from src.shared.core_utils import RANDOM_STATE
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores

# 2 x 3 x 2 x 3 x 2 x 2 = 144 combinations (trimmed from 2,700)
PARAM_GRID = {
    "bootstrap": [True, False],
    "max_depth": [20, 40, 60],        # 60 stands in for "unlimited" (see note below)
    "max_features": ["sqrt", 1.0],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 10],
    "n_estimators": [200, 400],
}


def main():
    stage = choose_stage("B")
    split = load_data(stage)

    def evaluate(params):
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(split.X_train, split.y_train.ravel())
        return validation_scores(split, model.predict(split.X_val))

    table, path = grid_search("random_forest", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_random_forest.py and run it.")


if __name__ == "__main__":
    main()