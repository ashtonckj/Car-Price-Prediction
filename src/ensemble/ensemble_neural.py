"""Ensemble methods: Level-1 Ridge stacking meta-learner and constrained optimal weighted blend.

Ensemble of PyTorch neural architectures (MLP, DCN-V2).
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
from src.tuned_models.run_neural import BEST_PARAMS, predict, train_dcnv2, train_mlp

STAGE = "b2_ensemble"     # results go to results/b2_ensemble/
BASE_STAGE = "b1_tuned"   # tuned parameters of the base models
MODEL_NAMES = ["MLP", "DCN-V2"]
FoldSplit = namedtuple("FoldSplit", ["X_train", "y_train", "X_val", "y_val"])


def main():
    stage = BASE_STAGE  # where the base models' BEST_PARAMS come from
    split = load_data(STAGE)

    X_tr, y_tr = split.X_train, split.y_train.ravel()
    X_va, y_va = split.X_val, split.y_val.ravel()
    X_te = split.X_test

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_models = len(MODEL_NAMES)

    oof_train = np.zeros((len(X_tr), n_models))
    val_preds = np.zeros((len(X_va), n_models))
    test_preds = np.zeros((len(X_te), n_models))

    mlp_params = BEST_PARAMS[stage]["MLP"]
    dcnv2_params = BEST_PARAMS[stage]["DCN-V2"]

    t_start = time.perf_counter()
    base_predict_time = 0.0  # time the base models spend predicting the test set

    for fold_idx, (tr_idx, oof_idx) in enumerate(kf.split(X_tr)):
        print(f"\n--- Fold {fold_idx + 1}/{kf.n_splits} ---")
        X_f_tr, y_f_tr = X_tr[tr_idx], y_tr[tr_idx]
        X_f_va = X_tr[oof_idx]

        fold_split = FoldSplit(
            X_f_tr,
            y_f_tr.reshape(-1, 1).astype("float32"),
            X_va,  # early stopping on the 10% validation rows, not on the out-of-fold rows: those predictions become
            split.y_val,  # the meta-features, so stopping on them would make the networks look better than they are
        )

        # 1. Fit MLP on fold training split (index 0)
        mlp_model, _ = train_mlp(mlp_params, fold_split)
        oof_train[oof_idx, 0] = np.asarray(predict(mlp_model, X_f_va)).ravel()
        val_preds[:, 0] += np.asarray(predict(mlp_model, X_va)).ravel() / kf.n_splits
        test_pred, seconds = timed(predict, mlp_model, X_te)
        base_predict_time += seconds
        test_preds[:, 0] += np.asarray(test_pred).ravel() / kf.n_splits

        # 2. Fit DCN-V2 on fold training split (index 1)
        dcnv2_model, _ = train_dcnv2(dcnv2_params, fold_split)
        oof_train[oof_idx, 1] = np.asarray(predict(dcnv2_model, X_f_va)).ravel()
        val_preds[:, 1] += np.asarray(predict(dcnv2_model, X_va)).ravel() / kf.n_splits
        test_pred, seconds = timed(predict, dcnv2_model, X_te)
        base_predict_time += seconds
        test_preds[:, 1] += np.asarray(test_pred).ravel() / kf.n_splits

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
    save_results(STAGE, "MLP_DCN", "Ensemble (MLP + DCN-V2)", results, split)


if __name__ == "__main__":
    main()
