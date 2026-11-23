"""Grid search for the distance-based models' hyperparameters (Part B)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.experiment import choose_models
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import tune_family, validation_scores
from src.tuned_models.run_distance import build_model, predict

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 8 x 3 x 5 = 120 combinations
    "KNN": ("distance_tuning_knn", {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "weights": ["uniform", "distance", "inv_sq"],   # inv_sq = 1 / distance^2
        "metric": ["manhattan", "minkowski_1.5", "euclidean", "minkowski_3", "cosine"],
    }),
     # 12 x 3 x 2 = 72 combinations
    "Radius Neighbours": ("distance_tuning_radius", {
        "radius_factor": [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5],
        "weights": ["uniform", "distance", "inv_sq"],
        "metric": ["manhattan", "euclidean"],
    }),
    # 4 x 11 x 2 = 88 combinations
    "Kernel Regression": ("distance_tuning_kernel", {
        "kernel": ["gaussian", "laplacian", "epanechnikov", "triangular"],
        "bandwidth_factor": [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0],
        "metric": ["manhattan", "euclidean"],
    }),
}


def main():
    stage = choose_stage("B")
    names = choose_models(list(PARAM_GRIDS), "tune")
    split = load_data(stage)

    def make_evaluate(name):
        def evaluate(params):
            model = build_model(name, params)
            model.fit(split.X_train, split.y_train.ravel())
            return validation_scores(split, predict(model, split.X_val))
        return evaluate

    tune_family(stage, names, PARAM_GRIDS, make_evaluate, "src/tuned_models/run_distance.py")


if __name__ == "__main__":
    main()