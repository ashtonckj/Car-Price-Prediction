"""Grid search for the KNN's hyperparameters on a Part B stage: every combination in PARAM_GRID is fitted once."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from src.tuned_models.run_knn import predict, train_model

# 8 x 3 x 5 = 120 combinations
PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
    "weights": ["uniform", "distance", "inv_sq"],   # inv_sq = 1 / distance^2
    "metric": ["manhattan", "minkowski_1.5", "euclidean", "minkowski_3", "cosine"],
}


def main():
    stage = choose_stage("B")
    split = load_data(stage)

    def evaluate(params):
        model, _ = train_model(params, split)
        return validation_scores(split, predict(model, split.X_val))

    table, path = grid_search("knn", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_knn.py and run it.")


if __name__ == "__main__":
    main()