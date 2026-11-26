"""Grid search for the linear models' hyperparameters (Part B)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import N_FOLDS, RESULTS_ROOT, compute_metrics
from src.shared.experiment import choose_models
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import tune_family, validation_scores
from src.tuned_models.run_linear import build_model, fit_predict, n_numeric_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Huber at a low max_iter is part of the search

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 25 x 2 x 4 x 2 = 400 combinations 
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
        "max_iter": [100, 200, 500, 1000], 
    }),
    # 7 x 2 x 2 = 28 combinations
    "Polynomial Regression": ("linear_tuning_polynomial", {
        "degree": [2, 3, 4, 5, 6, 7, 8], # 1 is for linear regression so not included; stops at 8 (9+ is too large to run)
        "interaction_only": [False, True],  # True = only cross terms (x1*x2), no squares (x1^2)
        "fit_intercept": [True, False],
    }),
}

# ---------- CV re-check of the best polynomial settings ----------
# The grid search scores each setting on ONE validation split. A higher-degree polynomial can look best there by luck
# (no extreme cars in the val rows) and still explode on other rows. So the TOP_N settings by val_RMSE are re-scored
# with the same 5-fold CV that run_linear.py reports, and the most reliable one is chosen on CV RMSE.
TOP_N = 5
CV_CSV = "linear_tuning_polynomial_cv"


def cv_check_polynomial(stage, n_numeric):
    """5-fold CV of the TOP_N polynomial settings of the tuning CSV. Writes results/<stage>/linear_tuning_polynomial_cv.csv
    (best = last row, lowest mean CV RMSE) and returns the best parameters."""
    csv_name, grid = PARAM_GRIDS["Polynomial Regression"]
    tuning = pd.read_csv(RESULTS_ROOT / stage / f"{csv_name}.csv")
    top = tuning[np.isfinite(tuning["val_RMSE"])].nsmallest(TOP_N, "val_RMSE")
    folds = [load_data(stage, fold) for fold in range(N_FOLDS)]  # same folds as src/shared/experiment.cross_validate

    rows = []
    for _, row in top.iterrows():
        params = {"degree": int(row["degree"]), "interaction_only": bool(row["interaction_only"]),
                  "fit_intercept": bool(row["fit_intercept"])}
        fold_metrics = []
        for split in folds:
            y_pred = fit_predict(split, "Polynomial Regression", params, n_numeric)["y_pred"]
            try:
                fold_metrics.append(compute_metrics(split.y_test, y_pred, split.inverse_transform_price))
            except ValueError:  # predictions overflowed to infinity
                fold_metrics.append({"RMSE": np.inf, "MAE": np.inf, "R2": -np.inf, "MAPE": np.inf})
        table = pd.DataFrame(fold_metrics)
        rows.append({**params, "val_RMSE": row["val_RMSE"],
                     **{f"cv_{m}_mean": table[m].mean() for m in table}, **{f"cv_{m}_std": table[m].std() for m in table},
                     "worst_fold_R2": table["R2"].min(),
                     **{f"fold{i + 1}_R2": r2 for i, r2 in enumerate(table["R2"])}})
        print(f"  {params}: val RMSE {row['val_RMSE']:,.0f} | CV RMSE {rows[-1]['cv_RMSE_mean']:,.0f} "
              f"+/- {rows[-1]['cv_RMSE_std']:,.0f} | CV R2 {rows[-1]['cv_R2_mean']:.4f} | worst fold R2 {rows[-1]['worst_fold_R2']:.4f}")

    result = pd.DataFrame(rows).sort_values("cv_RMSE_mean", ascending=False, kind="stable").round(4)
    path = RESULTS_ROOT / stage / f"{CV_CSV}.csv"
    result.to_csv(path, index=False)
    best = {k: result.iloc[-1][k].item() for k in grid}
    print(f"\nBest Polynomial Regression parameters by 5-fold CV: {best} | saved {path} (best = last row)")
    print(f'Put this in BEST_PARAMS in src/tuned_models/run_linear.py:\n        "Polynomial Regression": {best},')
    return best


def main():
    stage = choose_stage("B")
    names = choose_models(list(PARAM_GRIDS), "tune")
    split = load_data(stage)
    n_numeric = n_numeric_columns(split)

    def make_evaluate(name):
        def evaluate(params):
            model = build_model(name, params, n_numeric)
            model.fit(split.X_train, split.y_train.ravel())
            try:
                return validation_scores(split, model.predict(split.X_val))
            except ValueError:
                print(f"  {params}: predictions overflowed to infinity, recorded as the worst score")
                return {"val_RMSE": float("inf"), "val_MAE": float("inf"), "val_R2": float("-inf"), "val_MAPE": float("inf")}
        return evaluate

    tune_family(stage, names, PARAM_GRIDS, make_evaluate, "src/tuned_models/run_linear.py")

    if "Polynomial Regression" in names:
        print(f"\n===== Polynomial Regression: 5-fold CV re-check of the top {TOP_N} settings by val_RMSE =====")
        cv_check_polynomial(stage, n_numeric)


if __name__ == "__main__":
    main()
