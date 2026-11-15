"""Helpers for the tuning scripts: the grid-search loop and the results CSV in results/tuning/."""
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.shared.core_utils import RESULTS_ROOT, compute_metrics
from src.shared.experiment import timed

TUNING_DIR = RESULTS_ROOT / "tuning"


def save_tuning_csv(rows, model, stage, score_column):
    """Write results/tuning/<model>_<stage>.csv, one row per combination, worst score first (the best is the LAST row)."""
    table = pd.DataFrame(rows).sort_values(score_column, na_position="first", kind="stable").reset_index(drop=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    path = TUNING_DIR / f"{model}_{stage}.csv"
    table.to_csv(path, index=False)
    return table, path


def validation_scores(split, y_val_pred):
    """The validation-set scores saved for each combination (price in MAD; R2_log is the one to rank by)."""
    m = compute_metrics(split.y_val, y_val_pred, split.inverse_transform_price)
    return {"val_R2_log": round(m["R2_log"], 4), "val_R2": round(m["R2"], 4), "val_MAE": round(m["MAE"]), "val_MAPE": round(m["MAPE"], 2)}


def best_params(table, param_names):
    """The parameters of the last (best) row, as plain Python values (ints stay ints) ready to paste into BEST_PARAMS."""
    return table[param_names].iloc[[-1]].to_dict("records")[0]


def grid_search(model, stage, param_grid, evaluate, score_column="val_R2_log"):
    """Try every combination in param_grid. evaluate(params) returns a dict of scores that includes score_column.
    The CSV is rewritten after each combination, so stopping early still keeps the results so far."""
    combinations = list(ParameterGrid(param_grid))
    print(f"Grid: {len(combinations)} combinations")
    rows = []
    for number, params in enumerate(combinations, start=1):
        scores, seconds = timed(evaluate, params)
        rows.append({**{name: params[name] for name in param_grid}, **scores, "time_s": round(seconds, 1)})
        table, path = save_tuning_csv(rows, model, stage, score_column)
        print(f"[{number}/{len(combinations)}] {score_column} {scores[score_column]:.4f} ({seconds:.0f}s) | best so far {table[score_column].iloc[-1]:.4f} | {params}")
    return table, path
