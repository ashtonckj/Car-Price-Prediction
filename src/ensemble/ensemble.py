"""Ensemble methods: Level-1 Ridge stacking meta-learner and constrained optimal weighted blend.

Uses four tuned base models from Part B (Polynomial Regression, Gradient Boosting, Kernel Regression, MLP).
Out-of-fold predictions on the 70% training set form the meta-features; validation predictions are fold-averaged.
Finally evaluates on both validation and the 20% test set, and saves to results/b2_ensemble/.
The base models use their tuned BEST_PARAMS of BASE_STAGE (b1_tuned); no stage menu.
"""
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import save_results, timed
from src.shared.stages import load_data

# Tuned model builders / trainers (each family file has its own BEST_PARAMS, keyed by stage and then model name)
from src.tuned_models import run_distance, run_linear, run_neural, run_tree

STAGE = "b2_ensemble"     # results go to results/b2_ensemble/
BASE_STAGE = "b1_tuned"   # tuned parameters of the base models
MODEL_NAMES = ["Polynomial Regression", "Gradient Boosting", "Kernel Regression", "MLP"]  # the names used in the BEST_PARAMS
FoldSplit = namedtuple("FoldSplit", ["X_train", "y_train", "X_val", "y_val"])


def _build_sklearn_models(stage, n_numeric):
    """Return a list of (name, unfitted model) for the 3 non-neural base learners (all expose fit / predict)."""
    return [
        ("Polynomial Regression", run_linear.build_model("Polynomial Regression", run_linear.BEST_PARAMS[stage]["Polynomial Regression"], n_numeric)),
        ("Gradient Boosting", run_tree.build_model("Gradient Boosting", run_tree.BEST_PARAMS[stage]["Gradient Boosting"])),
        ("Kernel Regression", run_distance.build_model("Kernel Regression", run_distance.BEST_PARAMS[stage]["Kernel Regression"])),
    ]


def main():
    stage = BASE_STAGE  # where the base models' BEST_PARAMS come from
    split = load_data(STAGE)
    n_numeric = run_linear.n_numeric_columns()

    X_tr, y_tr = split.X_train, split.y_train.ravel()
    X_va, y_va = split.X_val, split.y_val.ravel()
    X_te = split.X_test

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_models = len(MODEL_NAMES)

    oof_train = np.zeros((len(X_tr), n_models))
    val_preds = np.zeros((len(X_va), n_models))
    test_preds = np.zeros((len(X_te), n_models))

    mlp_params = run_neural.BEST_PARAMS[stage]["MLP"]

    t_start = time.perf_counter()
    base_predict_time = 0.0  # time the base models spend predicting the test set

    for fold_idx, (tr_idx, oof_idx) in enumerate(kf.split(X_tr)):
        print(f"\n--- Fold {fold_idx + 1}/{kf.n_splits} ---")
        X_f_tr, y_f_tr = X_tr[tr_idx], y_tr[tr_idx]
        X_f_va = X_tr[oof_idx]

        # 1. Fit tuned non-neural models on fold training split
        sklearn_models = _build_sklearn_models(stage, n_numeric)
        for i, (name, model) in enumerate(sklearn_models):
            model.fit(X_f_tr, y_f_tr)
            oof_train[oof_idx, i] = model.predict(X_f_va).ravel()
            val_preds[:, i] += model.predict(X_va).ravel() / kf.n_splits
            test_pred, seconds = timed(model.predict, X_te)
            base_predict_time += seconds
            test_preds[:, i] += test_pred.ravel() / kf.n_splits

        # 2. Fit MLP on fold training split (always the last column)
        mlp_col = len(sklearn_models)
        fold_split = FoldSplit(
            X_f_tr,
            y_f_tr.reshape(-1, 1).astype("float32"),  # train_mlp expects a (n, 1) target
            X_va,  # early stopping on the 10% validation rows, not on the out-of-fold rows: those predictions become
            split.y_val,  # the meta-features, so stopping on them would make the networks look better than they are
        )
        mlp_model, _ = run_neural.train_mlp(mlp_params, fold_split)
        oof_train[oof_idx, mlp_col] = np.asarray(run_neural.predict(mlp_model, X_f_va)).ravel()
        val_preds[:, mlp_col] += np.asarray(run_neural.predict(mlp_model, X_va)).ravel() / kf.n_splits
        test_pred, seconds = timed(run_neural.predict, mlp_model, X_te)
        base_predict_time += seconds
        test_preds[:, mlp_col] += np.asarray(test_pred).ravel() / kf.n_splits

    # --- Method A: Level-1 Ridge Stacking Meta-Learner ---
    meta_ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    meta_ridge.fit(oof_train, y_tr)
    stack_val_pred = meta_ridge.predict(val_preds)
    stack_test_pred, stack_predict_time = timed(meta_ridge.predict, test_preds)

    # Duan's smearing bias correction (learned on validation set residuals)
    stack_val_pred_smeared = split.smear(stack_val_pred, stack_val_pred)
    stack_test_pred_smeared = split.smear(stack_val_pred, stack_test_pred)

    # --- Method B: Constrained Optimal Weighted Blend ---
    def blend_loss(weights):
        return root_mean_squared_error(y_tr, oof_train @ weights)

    res = minimize(
        blend_loss,
        x0=np.ones(n_models) / n_models,
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    blend_val_pred = val_preds @ res.x
    blend_test_pred, blend_predict_time = timed(np.dot, test_preds, res.x)
    # fit time = everything except predicting the test set; the base models are shared, so both methods get the same fit time
    fit_time = time.perf_counter() - t_start - base_predict_time - stack_predict_time - blend_predict_time

    # Duan's smearing bias correction
    blend_val_pred_smeared = split.smear(blend_val_pred, blend_val_pred)
    blend_test_pred_smeared = split.smear(blend_val_pred, blend_test_pred)

    # --- Results ---
    print("\nLearned Ridge Meta-Weights:", dict(zip(MODEL_NAMES, np.round(meta_ridge.coef_, 4))))
    print("Optimal Blend Weights:     ", dict(zip(MODEL_NAMES, np.round(res.x, 4))))
    print(f"Stacking Val RMSE (raw): {root_mean_squared_error(y_va, stack_val_pred):.5f}")
    print(f"Blending Val RMSE (raw): {root_mean_squared_error(y_va, blend_val_pred):.5f}")

    print("\n--- Validation Set Evaluation (Price in MAD) ---")
    evaluate("Stacking (Ridge) [Val]", split.y_val, stack_val_pred_smeared, split.inverse_transform_price)
    evaluate("Blending (Optimal) [Val]", split.y_val, blend_val_pred_smeared, split.inverse_transform_price)

    print("\n--- Test Set Evaluation (Price in MAD) ---")
    evaluate("Stacking (Ridge) [Test]", split.y_test, stack_test_pred_smeared, split.inverse_transform_price)
    evaluate("Blending (Optimal) [Test]", split.y_test, blend_test_pred_smeared, split.inverse_transform_price)

    # Save both models on test set (as expected by save_results)
    results = {
        "Stacking (Ridge)": {"y_pred": stack_test_pred_smeared, "fit_time": fit_time, "predict_time": base_predict_time + stack_predict_time},
        "Blending (Optimal)": {"y_pred": blend_test_pred_smeared, "fit_time": fit_time, "predict_time": base_predict_time + blend_predict_time},
    }
    # No CV for the ensembles: it would mean 5 x 5 fits per base model. The JSON has "cv": null.
    save_results(STAGE, "PR_GB_KR_MLP", "Ensemble (Polynomial + GB + Kernel Regression + MLP)", results, split)


if __name__ == "__main__":
    main()
