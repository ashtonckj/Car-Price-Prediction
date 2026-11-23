"""Grid search for the tree-based models' hyperparameters (Part B)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.experiment import choose_models
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import tune_family, validation_scores
from src.tuned_models.run_tree import build_model

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 6 x 5 x 5 x 2 x 2 x 4 = 2400 combinations
    "Decision Tree": ("tree_tuning_decision_tree", {
        "max_depth": [3, 5, 8, 12, 15, None],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": [None, "sqrt"],
        "criterion": ["squared_error", "absolute_error"],
        "ccp_alpha": [0.0, 0.001, 0.01, 0.05],
    }),
    # 6 x 3 x 4 x 3 x 3 = 648 combinations
    "Random Forest": ("tree_tuning_random_forest", {
        "max_depth": [10, 20, 30, 40, 50, None],
        "max_features": [1.0, "log2", "sqrt"],
        "min_samples_leaf": [1, 2, 4, 5],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [100, 300, 500],
    }),
    # 3 x 3 x 2 x 2 x 2 x 3 x 3 = 324 combinations
    "Gradient Boosting": ("tree_tuning_gradient_boosting", {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 8],
        "max_features": [1.0, "sqrt"],
        "min_samples_leaf": [1, 4],
        "min_samples_split": [2, 10],
        "n_estimators": [100, 300, 500],
        "subsample": [0.7, 0.85, 1.0],
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
            return validation_scores(split, model.predict(split.X_val))
        return evaluate

    tune_family(stage, names, PARAM_GRIDS, make_evaluate, "src/tuned_models/run_tree.py")


if __name__ == "__main__":
    main()
