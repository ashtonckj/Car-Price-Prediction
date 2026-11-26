"""Helpers for the tuning scripts: the grid-search loop and one results CSV per model, results/<stage>/<family>_tuning_<model>.csv"""
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.shared.core_utils import RESULTS_ROOT, compute_metrics
from src.shared.experiment import rounded, timed


def save_tuning_csv(rows, csv_name, stage, score_column, param_names, lower_is_better=True):
    """Write results/<stage>/<csv_name>.csv, one row per combination, worst score first (the best is the LAST row)."""
    params = pd.DataFrame([{name: row[name] for name in param_names} for row in rows], dtype=object)
    scores = pd.DataFrame([{k: v for k, v in row.items() if k not in param_names} for row in rows])
    table = pd.concat([params, scores], axis=1)
    table = table.sort_values(score_column, ascending=not lower_is_better, na_position="first", kind="stable").reset_index(drop=True)

    out_dir = RESULTS_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{csv_name}.csv"
    to_write = table.copy()
    to_write[param_names] = table[param_names].where(table[param_names].notna(), "None")
    to_write.to_csv(path, index=False)
    return table, path


def validation_scores(split, y_val_pred):
    """Every metric on the validation rows (price in MAD), as val_RMSE, val_MAE, val_R2, val_MAPE."""
    metrics = compute_metrics(split.y_val, y_val_pred, split.inverse_transform_price)
    return {f"val_{name}": value for name, value in rounded(metrics).items()}


def _clean_param_value(v):
    if pd.isna(v) or v in (None, "None"):
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.isdigit():
            return int(v)
        try:
            return float(v)
        except ValueError:
            return v
    return v


def best_params(table, param_names):
    """The parameters of the last (best) row, as plain Python values (ints stay ints) ready to paste into BEST_PARAMS."""
    raw = table[param_names].iloc[[-1]].to_dict("records")[0]
    return {k: _clean_param_value(v) for k, v in raw.items()}


def _normalize_param_value(val):
    if pd.isna(val) or val in (None, "None", ""):
        return None
    val_str = str(val).strip()
    try:
        return round(float(val_str), 8)
    except (ValueError, TypeError):
        return val_str


def _param_key(param_dict, param_names):
    return tuple((name, _normalize_param_value(param_dict.get(name))) for name in sorted(param_names))


def grid_search(csv_name, stage, param_grid, evaluate, score_column="val_RMSE", lower_is_better=True):
    """Try every combination in param_grid. evaluate(params) returns a dict of scores that includes score_column.
    The CSV is rewritten after each combination, so stopping early still keeps the results so far.
    If the CSV already exists (same parameters), its combinations are kept and only the remaining ones are run."""
    combinations = list(ParameterGrid(param_grid))
    param_names = list(param_grid)
    path = RESULTS_ROOT / stage / f"{csv_name}.csv"

    rows = []
    if path.exists():
        existing = pd.read_csv(path)
        if set(param_names + [score_column]) <= set(existing.columns):
            rows = existing.to_dict("records")
    completed_keys = {_param_key(r, param_names) for r in rows}

    remaining = [p for p in combinations if _param_key(p, param_names) not in completed_keys]

    print(f"Grid: {len(combinations)} total combinations")
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} already completed, {len(remaining)} remaining.")

    if not remaining:
        return save_tuning_csv(rows, csv_name, stage, score_column, param_names, lower_is_better)

    for idx, params in enumerate(remaining, start=1):
        total_done = len(completed_keys) + idx
        scores, seconds = timed(evaluate, params)
        rows.append({**{name: params[name] for name in param_grid}, **scores, "time_s": round(seconds, 1)})
        table, path = save_tuning_csv(rows, csv_name, stage, score_column, param_names, lower_is_better)
        best_score = table[score_column].iloc[-1]
        print(f"[{total_done}/{len(combinations)}] {score_column} {scores[score_column]:,.2f} ({seconds:.0f}s) | best so far {best_score:,.2f} | {params}")

    return table, path


def tune_family(stage, names, param_grids, make_evaluate, run_file):
    """Tune each chosen model of a family on its own grid, one CSV per model.
    param_grids:   {model name: (csv name, parameter grid)}
    make_evaluate: make_evaluate(model name) returns evaluate(params) -> dict of validation scores
    run_file:      only used in the final hint, e.g. "src/tuned_models/run_linear.py" """
    found = {}
    for name in names:
        csv_name, param_grid = param_grids[name]
        print(f"\n===== {name} =====")
        table, path = grid_search(csv_name, stage, param_grid, make_evaluate(name))
        found[name] = best_params(table, list(param_grid))
        print(f"Best {name} parameters: {found[name]} | saved {path} (best = last row)")

    print(f"\nBest parameters found. If you agree, put them in BEST_PARAMS in {run_file} and run it:")
    for name, params in found.items():
        print(f'        "{name}": {params},')
    return found
