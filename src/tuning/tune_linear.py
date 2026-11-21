"""Grid search for the linear models' hyperparameters on a Part B stage: every combination in each PARAM_GRIDS entry is
trained once. The best combination is the one with the highest validation R2."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RESULTS_ROOT
from src.shared.experiment import timed
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, save_tuning_csv, validation_scores
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
    # 8 x 5 x 4 = 160 combinations (slow: about 25 s per 100 iterations, so up to ~4 min at max_iter=1000)
    "Huber": ("huber", {
        "epsilon": [1.1, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "max_iter": [100, 200, 500, 1000],  # shows whether more L-BFGS iterations help (earlier: 100 was as good as 1000)
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


def grid_search_resume(csv_name, stage, param_grid, evaluate, score_column="val_R2"):
    """Same as the shared grid_search (higher score_column = better), but combinations already in
    results/<stage>/<csv_name>_tuning.csv are reused instead of trained again, so a long search (Huber: hours) can be
    stopped and restarted. Delete the CSV first if build_model() has changed since it was written."""
    names = list(param_grid)
    path = RESULTS_ROOT / stage / f"{csv_name}_tuning.csv"
    done = {}
    if path.exists():
        old = pd.read_csv(path)
        if set(names) <= set(old.columns):
            done = {tuple(row[n] for n in names): row for row in old.to_dict("records")}

    combinations = list(ParameterGrid(param_grid))
    reused = sum(tuple(p[n] for n in names) in done for p in combinations)
    print(f"Grid: {len(combinations)} combinations ({reused} already in {path.name}, {len(combinations) - reused} to train)")
    rows = []
    for number, params in enumerate(combinations, start=1):
        key = tuple(params[n] for n in names)
        if key in done:
            rows.append(done[key])
            continue
        scores, seconds = timed(evaluate, params)
        rows.append({**{n: params[n] for n in names}, **scores, "time_s": round(seconds, 1)})
        table, path = save_tuning_csv(rows, csv_name, stage, score_column, lower_is_better=False)
        print(f"[{number}/{len(combinations)}] {score_column} {scores[score_column]:,.4f} ({seconds:.0f}s) | best so far {table[score_column].iloc[-1]:,.4f} | {params}")
    table, path = save_tuning_csv(rows, csv_name, stage, score_column, lower_is_better=False)
    return table, path


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

        # combinations already in the CSV are not trained again (Huber's 160-combination grid takes hours)
        table, path = grid_search_resume(csv_name, stage, param_grid, evaluate, score_column="val_R2")
        found[name] = best_params(table, list(param_grid))
        print(f"Best {name} parameters: {found[name]} | saved {path} (best = last row)")

    print("\nBest parameters:")
    for name, params in found.items():
        print(f'        "{name}": {params},')


if __name__ == "__main__":
    main()
