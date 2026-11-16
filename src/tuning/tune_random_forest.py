"""Tune a Random Forest regressor on a2_preprocessed using 5-fold grid search on the 70/10/20 split, then evaluate the best config.

Save as src/tuning/tune_random_forest.py (ROOT_DIR assumes src/<folder>/file.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RandomizedSearchCV

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import N_FOLDS, RANDOM_STATE, RESULTS_ROOT, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import load_data

STAGE = "a2_preprocessed"
SEARCH_MODE = "grid"  # "grid" for full 5-fold GridSearchCV, or "random" for RandomizedSearchCV
N_ITER = 50           # Number of sampled parameter combinations if SEARCH_MODE == "random"

# Parameter grid specified for Random Forest Regressor
PARAM_GRID = {
    "bootstrap": [True, False],
    "max_depth": [10, 20, 30, 40, 50, None],
    "max_features": [1.0, "log2", "sqrt"],
    "min_samples_leaf": [1, 2, 4, 5, 10],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [100, 200, 300, 400, 500],
}


def train_model(params, split):
    """Train Random Forest with given params on split.X_train. Returns (fitted model, val MSE)."""
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(split.X_train, split.y_train.ravel())
    val_pred = model.predict(split.X_val)
    val_mse = float(np.mean((val_pred - split.y_val.ravel()) ** 2))
    return model, val_mse


def tune(X_dev, y_dev, mode=SEARCH_MODE, n_iter=N_ITER):
    """5-fold cross-validation search on the 80% development set (train + val)."""
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base_estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    if mode == "random":
        print(f"Running 5-Fold RandomizedSearchCV on 80% dev set ({n_iter} configurations)...")
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=PARAM_GRID,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )
    else:
        total_combos = len(list(ParameterGrid(PARAM_GRID)))
        print(f"Running 5-Fold GridSearchCV on 80% dev set ({total_combos} combinations x {N_FOLDS} folds = {total_combos * N_FOLDS} fits)...")
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=PARAM_GRID,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )
    _, search_time = timed(search.fit, X_dev, y_dev)
    print(f"Search completed in {search_time:.1f}s ({search_time / 60:.1f}m)")
    trials = []
    cv_res = search.cv_results_
    for i in range(len(cv_res["params"])):
        trials.append({
            "params": cv_res["params"][i],
            "cv_mse": round(-float(cv_res["mean_test_score"][i]), 5),
            "std_mse": round(float(cv_res["std_test_score"][i]), 5),
            "rank": int(cv_res["rank_test_score"][i]),
            "time_s": round(float(cv_res["mean_fit_time"][i]), 2),
        })
    trials.sort(key=lambda tr: tr["rank"])
    return search.best_params_, trials, search.best_estimator_


def main():
    mode = SEARCH_MODE
    n_iter = N_ITER

    # Optional CLI overrides: py -m src.tuning.tune_random_forest --random 30
    if "--random" in sys.argv:
        mode = "random"
        idx = sys.argv.index("--random")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            n_iter = int(sys.argv[idx + 1])

    split = load_data(STAGE)
    print(f"Loaded {STAGE} (70/10/20 split): train {len(split.X_train)}, val {len(split.X_val)}, test {len(split.X_test)}")

    # Combine 70% train and 10% val into an 80% development set
    X_dev = np.vstack([split.X_train, split.X_val])
    y_dev = np.concatenate([split.y_train.ravel(), split.y_val.ravel()])
    print(f"Loaded {STAGE}: 80% Dev (train+val) {len(X_dev)} samples, 20% Test {len(split.X_test)} samples")

    best_params, trials, best_estimator = tune(X_dev, y_dev, mode=mode, n_iter=n_iter)
    print(f"\nBest params: {best_params}")

    out_dir = RESULTS_ROOT / "tuning" / "random_forest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_file = out_dir / "random_forest_tuning.json"
    tuning_file.write_text(json.dumps({"best_params": best_params, "trials": trials}, indent=2))
    print(f"Saved tuning log to {tuning_file}")

    def fit_predict(s):
        (model, _), fit_time = timed(train_model, best_params, s)
        y_pred, predict_time = timed(model.predict, s.X_test)
        return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}

    name = "Random Forest (tuned)"
    print(f"\n===== {name} =====")
    final_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    _, fit_time = timed(final_model.fit, X_dev, y_dev)
    y_pred, predict_time = timed(final_model.predict, split.X_test)
    
    results = {name: fit_predict(split)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)


if __name__ == "__main__":
    main()
