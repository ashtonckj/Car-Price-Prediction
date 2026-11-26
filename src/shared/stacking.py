"""Shared code of the ensemble scripts (src/ensemble/, Part B step 2): Ridge stacking and an optimal weighted blend.

Out-of-fold predictions of the base models on the 70% training rows are the meta-features; their validation and test
predictions are averaged over the folds. The base models use their tuned BEST_PARAMS of BASE_STAGE; results go to
results/b2_ensemble/.
"""
import time
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from src.shared.core_utils import N_FOLDS, RANDOM_STATE, evaluate
from src.shared.experiment import save_results, timed
from src.tuned_models import run_neural

STAGE = "b2_ensemble"
BASE_STAGE = "b1_tuned"
FoldSplit = namedtuple("FoldSplit", ["X_train", "y_train", "X_val", "y_val"])


def sklearn_learner(make_model):
    """Base learner from a model factory: fit(X, y) trains a new model and returns its predict function."""
    def fit(X, y):
        model = make_model().fit(X, y)
        return lambda X_new: np.asarray(model.predict(X_new)).ravel()
    return fit


def network_learner(train, params, split):
    """Base learner from run_neural.train_mlp / train_dcnv2. Early stopping uses the 10% validation rows, not the
    out-of-fold rows: those predictions become the meta-features, so stopping on them would flatter the networks."""
    def fit(X, y):
        model, _ = train(params, FoldSplit(X, y.reshape(-1, 1).astype("float32"), split.X_val, split.y_val))
        return lambda X_new: np.asarray(run_neural.predict(model, X_new)).ravel()
    return fit


def stack_and_blend(split, learners, key, title):
    """learners: {model name: fit(X, y) -> predict}. Saves results/b2_ensemble/<key>_*."""
    names = list(learners)
    X_tr, y_tr = split.X_train, split.y_train.ravel()
    oof_train = np.zeros((len(X_tr), len(names)))
    val_preds = np.zeros((len(split.X_val), len(names)))
    test_preds = np.zeros((len(split.X_test), len(names)))

    t_start = time.perf_counter()
    base_predict_time = 0.0  # time the base models spend predicting the test set
    folds = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_tr)
    for fold, (fit_idx, oof_idx) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        for i, fit in enumerate(learners.values()):
            predict = fit(X_tr[fit_idx], y_tr[fit_idx])
            oof_train[oof_idx, i] = predict(X_tr[oof_idx])
            val_preds[:, i] += predict(split.X_val) / N_FOLDS
            test_pred, seconds = timed(predict, split.X_test)
            base_predict_time += seconds
            test_preds[:, i] += test_pred / N_FOLDS

    # Method A: Ridge stacking meta-learner
    meta_ridge = RidgeCV(alphas=np.logspace(-2, 3, 50)).fit(oof_train, y_tr)
    stack_val_pred = meta_ridge.predict(val_preds)
    stack_test_pred, stack_predict_time = timed(meta_ridge.predict, test_preds)

    # Method B: weights >= 0 summing to 1 that minimise the out-of-fold RMSE
    res = minimize(
        lambda w: root_mean_squared_error(y_tr, oof_train @ w),
        x0=np.ones(len(names)) / len(names),
        bounds=[(0, 1)] * len(names),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    blend_val_pred = val_preds @ res.x
    blend_test_pred, blend_predict_time = timed(np.dot, test_preds, res.x)
    # fit time = everything except predicting the test set; the base models are shared, so both methods get the same fit time
    fit_time = time.perf_counter() - t_start - base_predict_time - stack_predict_time - blend_predict_time

    print("\nLearned Ridge Meta-Weights:", dict(zip(names, np.round(meta_ridge.coef_, 4))))
    print("Optimal Blend Weights:     ", dict(zip(names, np.round(res.x, 4))))
    print(f"Stacking Val RMSE (raw): {root_mean_squared_error(split.y_val.ravel(), stack_val_pred):.5f}")
    print(f"Blending Val RMSE (raw): {root_mean_squared_error(split.y_val.ravel(), blend_val_pred):.5f}")

    # Duan's smearing, learned on the validation residuals
    methods = {"Stacking (Ridge)": (stack_val_pred, stack_test_pred, stack_predict_time),
               "Blending (Optimal)": (blend_val_pred, blend_test_pred, blend_predict_time)}
    print("\n--- Validation Set Evaluation (Price in MAD) ---")
    for method, (val_pred, _, _) in methods.items():
        evaluate(f"{method} [Val]", split.y_val, split.smear(val_pred, val_pred), split.inverse_transform_price)
    print("\n--- Test Set Evaluation (Price in MAD) ---")
    results = {}
    for method, (val_pred, test_pred, predict_time) in methods.items():
        y_pred = split.smear(val_pred, test_pred)
        evaluate(f"{method} [Test]", split.y_test, y_pred, split.inverse_transform_price)
        results[method] = {"y_pred": y_pred, "fit_time": fit_time, "predict_time": base_predict_time + predict_time}

    # No CV for the ensembles: it would mean 5 x 5 fits per base model. The JSON has "cv": null.
    save_results(STAGE, key, title, results, split)
