"""Timing, 5-fold cross-validation, and saving results to results/<stage>/<family>_results.json."""
import json
import time

import numpy as np
import pandas as pd

from src.shared.core_utils import N_FOLDS, RESULTS_ROOT, compute_metrics
from src.shared.plots import plot_family
from src.shared.stages import STAGES, load_data

METRICS = ["RMSE", "MAE", "R2", "MAPE", "R2_log"]


def timed(function, *args, **kwargs):
    """Call function(*args, **kwargs) and return (its result, seconds it took)."""
    t_start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - t_start


def cross_validate(stage, fit_predict):
    """5-fold CV on the 70% train part. fit_predict(split) must train on split.X_train and return {"y_pred": ...}
    for split.X_test. Returns each metric's mean and std over the folds (the std shows how stable the model is)."""
    fold_metrics = []
    for fold in range(N_FOLDS):
        split = load_data(stage, fold)
        y_pred = fit_predict(split)["y_pred"]
        fold_metrics.append(compute_metrics(split.y_test, y_pred, split.inverse_transform_price))
        print(f"  CV fold {fold + 1}: R2 {fold_metrics[-1]['R2']:.4f} | R2 log {fold_metrics[-1]['R2_log']:.4f} | MAE {fold_metrics[-1]['MAE']:,.0f}")
    table = pd.DataFrame(fold_metrics)
    cv = {"mean": table.mean().to_dict(), "std": table.std().to_dict()}
    print(f"  CV mean +/- std: R2 {cv['mean']['R2']:.4f} +/- {cv['std']['R2']:.4f} | R2 log {cv['mean']['R2_log']:.4f} +/- {cv['std']['R2_log']:.4f}")
    return cv


def rounded(values, digits=4):
    return {name: round(value, digits) for name, value in values.items()}


def save_results(stage, key, title, results, split, cv_results=None):
    """Write results/<stage>/<key>_results.json and <key>_pred_vs_actual.png.

    results:    {model name: {"y_pred", "fit_time", "predict_time"}} on the final 20% test set
    cv_results: {model name: output of cross_validate()} (optional)
    """
    cv_results = cv_results or {}
    models, predicted_prices = {}, {}
    for name, r in results.items():
        y_pred = np.asarray(r["y_pred"]).reshape(-1)  # sklearn gives (n,), torch gives (n, 1)
        predicted_prices[name] = split.inverse_transform_price(y_pred)
        cv = cv_results.get(name)
        models[name] = {
            "metrics": rounded(compute_metrics(split.y_test, y_pred, split.inverse_transform_price)),
            "cv": cv and {"mean": rounded(cv["mean"]), "std": rounded(cv["std"])},
            "fit_time_s": round(r["fit_time"], 2),
            "predict_time_s": round(r["predict_time"], 2),
        }
    saved = {
        "stage": stage,
        "stage_title": STAGES[stage]["title"],
        "family": title,
        "n_features": int(split.X_train.shape[1]),
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "models": models,
    }
    out_dir = RESULTS_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_family(f"{title} - {STAGES[stage]['title']}", split.inverse_transform_price(split.y_test),
                predicted_prices, models, out_dir / f"{key}_pred_vs_actual.png")
    (out_dir / f"{key}_results.json").write_text(json.dumps(saved, indent=2))
    print(f"\nSaved {key}_results.json and {key}_pred_vs_actual.png in {out_dir}")


def load_stage_results(stage):
    """One row per model for one stage (every family that has been run), best test R2_log first."""
    rows = []
    for path in sorted((RESULTS_ROOT / stage).glob("*_results.json")):
        family = json.loads(path.read_text())
        for name, r in family["models"].items():
            row = {"Stage": stage, "Family": family["family"], "Model": name, **r["metrics"]}
            if r.get("cv"):
                row.update({f"CV_{m}_mean": r["cv"]["mean"][m] for m in METRICS})
                row.update({f"CV_{m}_std": r["cv"]["std"][m] for m in METRICS})
            row.update({"Fit_time_s": r["fit_time_s"], "Predict_time_s": r["predict_time_s"]})
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("R2_log", ascending=False).reset_index(drop=True)
