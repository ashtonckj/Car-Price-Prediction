"""Ensemble methods: Level-1 Ridge stacking meta-learner and constrained optimal weighted blend.

Ensemble of PyTorch neural architectures (MLP, DCN-V2).
Out-of-fold predictions on the 70% training set form the meta-features; validation predictions are fold-averaged.
Finally evaluates on both validation and the 20% test set, and saves results following the project standard.
Run it and type a stage number.
"""
import sys
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
from src.shared.experiment import save_results
from src.shared.stages import choose_stage, load_data
from src.tuned_models.run_neural import BEST_PARAMS, predict, train_dcnv2, train_mlp

MODEL_NAMES = ["MLP", "DCN-V2"]
FoldSplit = namedtuple("FoldSplit", ["X_train", "y_train", "X_val", "y_val"])


def main():
    stage = choose_stage("B")
    split = load_data(stage)

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

    for fold_idx, (tr_idx, oof_idx) in enumerate(kf.split(X_tr)):
        print(f"\n--- Fold {fold_idx + 1}/{kf.n_splits} ---")
        X_f_tr, y_f_tr = X_tr[tr_idx], y_tr[tr_idx]
        X_f_va, y_f_va = X_tr[oof_idx], y_tr[oof_idx]

        fold_split = FoldSplit(
            X_f_tr,
            y_f_tr.reshape(-1, 1).astype("float32"),
            X_f_va,
            y_f_va.reshape(-1, 1).astype("float32"),
        )

        # 1. Fit MLP on fold training split (index 0)
        mlp_model, _ = train_mlp(mlp_params, fold_split)
        oof_train[oof_idx, 0] = np.asarray(predict(mlp_model, X_f_va)).ravel()
        val_preds[:, 0] += np.asarray(predict(mlp_model, X_va)).ravel() / kf.n_splits
        test_preds[:, 0] += np.asarray(predict(mlp_model, X_te)).ravel() / kf.n_splits

        # 2. Fit DCN-V2 on fold training split (index 1)
        dcnv2_model, _ = train_dcnv2(dcnv2_params, fold_split)
        oof_train[oof_idx, 1] = np.asarray(predict(dcnv2_model, X_f_va)).ravel()
        val_preds[:, 1] += np.asarray(predict(dcnv2_model, X_va)).ravel() / kf.n_splits
        test_preds[:, 1] += np.asarray(predict(dcnv2_model, X_te)).ravel() / kf.n_splits

    # --- Method A: Level-1 Ridge Stacking Meta-Learner ---
    meta_ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    meta_ridge.fit(oof_train, y_tr)
    stack_val_pred = meta_ridge.predict(val_preds)
    stack_test_pred = meta_ridge.predict(test_preds)

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
    blend_test_pred = test_preds @ res.x

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
        "Stacking (Ridge)": {"y_pred": stack_test_pred_smeared, "fit_time": 0.0, "predict_time": 0.0},
        "Blending (Optimal)": {"y_pred": blend_test_pred_smeared, "fit_time": 0.0, "predict_time": 0.0},
    }
    save_results(stage, "MLP_DCN", "Ensemble", results, split)


if __name__ == "__main__":
    main()
