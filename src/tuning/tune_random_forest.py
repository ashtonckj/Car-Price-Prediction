"""Grid search for the MLP's hyperparameters on a Part B stage: every combination in PARAM_GRID is trained once."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from sklearn.ensemble import RandomForestRegressor
from src.shared.tuning import grid_search, validation_scores

PARAM_GRID = {
    "max_depth": [10, 20, 30, 40, 50, None],
    "max_features": [1.0, "log2", "sqrt"],
    "min_samples_leaf": [1, 2, 4, 5],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [100, 300, 500],
}


def main():
    stage = choose_stage("B")
    split = load_data(stage)

    def evaluate(params):
        rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1 )
        rf.fit(split.X_train, split.y_train.ravel())
        y_pred = rf.predict(split.X_val)
        return validation_scores(split, y_pred)

    table, path = grid_search("random_forest", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_random_forest.py and run it.")


if __name__ == "__main__":
    main()
