"""Grid search for the linear models' hyperparameters on a Part B stage: every combination in each PARAM_GRIDS entry is
trained once. The best combination is the one with the highest validation R2."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from src.tuned_models.run_linear import build_model, n_numeric_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Huber at a low max_iter is part of the search

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 25 x 2 x 4 x 2 = 400 combinations (fast: about 1-5 s each)
    "Ridge": ("ridge", {
        "alpha": [float(a) for a in np.logspace(-3, 3, 25)],  # 0.001 ... 1000
        "fit_intercept": [True, False],
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg"],
        "tol": [1e-4, 1e-3],  # only used by lsqr / sparse_cg
    }),
    # 6 x 4 x 2 = 48 combinations (slow: about 20-50 s each)
    "Huber": ("huber", {
        "epsilon": [1.1, 1.35, 1.5, 2.0, 2.5, 3.0],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "max_iter": [100, 200],  # a larger 1000-iteration search was no better than 100
    }),
    # 1 x 2 x 2 = 4 combinations. Degree 2 only: unpenalised degree 3/4 explode on the few rows with extreme
    # mileage / fiscal power (degree 3 test R2 of about -1e7, degree 4 explodes even on validation)
    "Polynomial Regression": ("polynomial", {
        "degree": [2],
        "interaction_only": [False, True],  # True = only cross terms (x1*x2), no squares (x1^2)
        "fit_intercept": [True, False],
    }),
}


def choose_models():
    names = list(PARAM_GRIDS)
    print("Which linear models do you want to tune?")
    for number, name in enumerate(names, start=1):
        print(f"  {number} = {name}")
    choice = input("Type numbers separated by commas (Enter = all): ").strip()
    if not choice:
        return names
    picked = [c.strip() for c in choice.split(",")]
    if not all(c.isdigit() and 1 <= int(c) <= len(names) for c in picked):
        sys.exit(f"'{choice}' is not an option.")
    return [names[int(c) - 1] for c in picked]


def main():
    stage = choose_stage("B")
    models = choose_models()
    split = load_data(stage)
    n_numeric = n_numeric_columns()

    found = {}
    for name in models:
        csv_name, param_grid = PARAM_GRIDS[name]
        print(f"\n===== {name} =====")

        def evaluate(params):
            model = build_model(name, params, n_numeric)
            model.fit(split.X_train, split.y_train.ravel())
            return validation_scores(split, model.predict(split.X_val))

        table, path = grid_search(csv_name, stage, param_grid, evaluate, score_column="val_R2", lower_is_better=False)
        found[name] = best_params(table, list(param_grid))
        print(f"Best {name} parameters: {found[name]} | saved {path} (best = last row)")

    print("\nBest parameters:")
    for name, params in found.items():
        print(f'        "{name}": {params},')
    print("Next: put them in BEST_PARAMS in src/tuned_models/run_linear.py and run it.")


if __name__ == "__main__":
    main()
