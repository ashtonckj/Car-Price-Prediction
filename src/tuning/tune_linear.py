"""Grid search for the linear models' hyperparameters (Part B)."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.experiment import choose_models
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import tune_family, validation_scores
from src.tuned_models.run_linear import build_model, n_numeric_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Huber at a low max_iter is part of the search

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 25 x 2 x 4 x 2 = 400 combinations (fast: about 1-5 s each)
    "Ridge": ("linear_tuning_ridge", {
        "alpha": [float(a) for a in np.logspace(-3, 3, 25)],  # 0.001 ... 1000
        "fit_intercept": [True, False],
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg"],
        "tol": [1e-4, 1e-3],  # only used by lsqr / sparse_cg
    }),
    # 8 x 5 x 4 = 160 combinations (slow: about 25 s per 100 iterations, so up to ~4 min at max_iter=1000)
    "Huber": ("linear_tuning_huber", {
        "epsilon": [1.1, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "max_iter": [100, 200, 500, 1000],  # shows whether more L-BFGS iterations help (earlier: 100 was as good as 1000)
    }),
    # 1 x 2 x 2 = 4 combinations. Degree 2 only: unpenalised degree 3/4 explode on the few rows with extreme
    # mileage / fiscal power (degree 3 test R2 of about -1e7, degree 4 explodes even on validation)
    "Polynomial Regression": ("linear_tuning_polynomial", {
        "degree": [2],
        "interaction_only": [False, True],  # True = only cross terms (x1*x2), no squares (x1^2)
        "fit_intercept": [True, False],
    }),
}


def main():
    stage = choose_stage("B")
    names = choose_models(list(PARAM_GRIDS), "tune")
    split = load_data(stage)
    n_numeric = n_numeric_columns()

    def make_evaluate(name):
        def evaluate(params):
            model = build_model(name, params, n_numeric)
            model.fit(split.X_train, split.y_train.ravel())
            return validation_scores(split, model.predict(split.X_val))
        return evaluate

    tune_family(stage, names, PARAM_GRIDS, make_evaluate, "src/tuned_models/run_linear.py")


if __name__ == "__main__":
    main()
