"""Timing, 5-fold cross-validation, the model menu, the family run loop, and saving results to results/<stage>/<family>_results.json."""
import json
import sys
import time

import numpy as np
import pandas as pd

from src.shared.core_utils import N_FOLDS, RESULTS_ROOT, compute_metrics, evaluate
from src.shared.plots import plot_family
from src.shared.saved_models import save_model
from src.shared.stages import STAGES, load_data

METRICS = ["RMSE", "MAE", "R2", "MAPE"]


def timed(function, *args, **kwargs):
    """Call function(*args, **kwargs) and return (its result, seconds it took)."""
    t_start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - t_start


def cross_validate(stage, run):
    """5-fold CV on the 70% train part. run(split) must train on split.X_train and return {"y_pred": ...}
    for split.X_test. Returns each metric's mean and std over the folds (the std shows how stable the model is)."""
    fold_metrics = []
    for fold in range(N_FOLDS):
        split = load_data(stage, fold)
        y_pred = run(split)["y_pred"]
        fold_metrics.append(compute_metrics(split.y_test, y_pred, split.inverse_transform_price))
        print(f"  CV fold {fold + 1}: R2 {fold_metrics[-1]['R2']:.4f} | MAE {fold_metrics[-1]['MAE']:,.0f}")
    table = pd.DataFrame(fold_metrics)
    cv = {"mean": table.mean().to_dict(), "std": table.std().to_dict()}
    print(f"  CV mean +/- std: R2 {cv['mean']['R2']:.4f} +/- {cv['std']['R2']:.4f}")
    return cv


def rounded(values, digits=4):
    return {name: round(value, digits) for name, value in values.items()}


def choose_models(names, action="run", what="models"):
    """Numbered menu so one model of a family can be run or tuned on its own. Enter = every model.
    Returns the chosen names in menu order."""
    print(f"Which {what} do you want to {action}?")
    for number, name in enumerate(names, start=1):
        print(f"  {number} = {name}")
    choice = input("Type numbers separated by commas (Enter = all): ").strip()
    if not choice:
        return list(names)
    picked = [c.strip() for c in choice.split(",")]
    if not all(c.isdigit() and 1 <= int(c) <= len(names) for c in picked):
        sys.exit(f"'{choice}' is not an option.")
    numbers = sorted({int(c) for c in picked})
    return [names[n - 1] for n in numbers]


def save_results(stage, key, title, results, split, cv_results=None, order=None):
    """Write results/<stage>/<key>_results.json, <key>_predictions.npz and <key>_pred_vs_actual.png.

    The files MERGE with what earlier runs of this family saved: the models in `results` replace their own old
    entries, every other model already saved is kept. So one model can be rerun without rerunning the rest.

    results:    {model name: {"y_pred", "fit_time", "predict_time"}} on the final 20% test set
    cv_results: {model name: output of cross_validate()} (optional)
    order:      model names in the order they should appear (optional; others follow)
    """
    cv_results = cv_results or {}
    out_dir = RESULTS_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{key}_results.json"
    preds_path = out_dir / f"{key}_predictions.npz"  # test predictions in MAD, so the merged graph can be redrawn
    y_true_price = split.inverse_transform_price(split.y_test)

    # what earlier runs of this family saved
    models, predicted_prices = {}, {}
    if json_path.exists():
        models = json.loads(json_path.read_text())["models"]
    if preds_path.exists():
        with np.load(preds_path) as saved_predictions:
            for name in saved_predictions.files:
                if name in models and len(saved_predictions[name]) == len(y_true_price):
                    predicted_prices[name] = saved_predictions[name]

    # the models that were just run replace their old entries
    for name, r in results.items():
        y_pred = np.asarray(r["y_pred"]).reshape(-1)  # sklearn gives (n,), torch gives (n, 1)
        predicted_prices[name] = np.asarray(split.inverse_transform_price(y_pred)).reshape(-1)
        cv = cv_results.get(name)
        models[name] = {
            "metrics": rounded(compute_metrics(split.y_test, y_pred, split.inverse_transform_price)),
            "cv": cv and {"mean": rounded(cv["mean"]), "std": rounded(cv["std"])},
            "fit_time_s": round(r["fit_time"], 2),
            "predict_time_s": round(r["predict_time"], 2),
        }

    first = [name for name in (order or []) if name in models]
    models = {name: models[name] for name in first + [name for name in models if name not in first]}

    saved = {
        "stage": stage,
        "stage_title": STAGES[stage]["title"],
        "family": title,
        "n_features": int(split.X_train.shape[1]),
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "models": models,
    }

    # the graph shows every model that has saved predictions
    plotted = {name: m for name, m in models.items() if name in predicted_prices}
    not_plotted = [name for name in models if name not in plotted]
    if not_plotted:
        print(f"\nNot in the graph (no saved predictions yet, run them once more to add them): {', '.join(not_plotted)}")
    plot_family(f"{title} - {STAGES[stage]['title']}", y_true_price, predicted_prices, plotted,
                out_dir / f"{key}_pred_vs_actual.png")
    np.savez_compressed(preds_path, **{name: predicted_prices[name] for name in plotted})
    json_path.write_text(json.dumps(saved, indent=2))
    print(f"\nSaved {key}_results.json and {key}_pred_vs_actual.png in {out_dir} "
          f"({len(results)} model(s) updated, {len(models)} in the file)")


def fit_predict(make_model, split):
    """Train a new sklearn-style model on split.X_train and predict split.X_test (log-price bias corrected)."""
    model = make_model()
    _, fit_time = timed(model.fit, split.X_train, split.y_train.ravel())
    y_pred, predict_time = timed(model.predict, split.X_test)
    y_pred = split.smear(model.predict(split.X_val), y_pred)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def run_family(stage, split, key, title, models, tuned_params=None):
    """Every model of a family: 5-fold CV, then trained on the final split, test results printed and saved.
    models:       {model name: fit_predict(split) -> {"model", "y_pred", "fit_time", "predict_time"}}
    tuned_params: BEST_PARAMS[stage] of a Part B run: printed, used for the model order in the JSON, and every
                  trained model is saved to saved_models/ (Part A runs save results only)."""
    results, cv_results = {}, {}
    for name, run in models.items():
        print(f"\n===== {name} (tuned) | parameters: {tuned_params[name]} =====" if tuned_params else f"\n===== {name} =====")
        cv_results[name] = cross_validate(stage, run)
        results[name] = run(split)
        if tuned_params:
            save_model(stage, key, name, results[name]["model"], split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)
    save_results(stage, key, title, results, split, cv_results, order=tuned_params)


def load_stage_results(stage):
    """One row per model for one stage (every family that has been run), best test R2 first."""
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
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
