"""Helpers for the tuning scripts: the grid-search loop and the results CSV in results/tuning/.
Combinations are judged by RMSE on the price in MAD: the lowest RMSE is the best."""
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.shared.core_utils import RESULTS_ROOT, compute_metrics
from src.shared.experiment import METRICS, rounded, timed

TUNING_DIR = RESULTS_ROOT / "tuning"


def save_tuning_csv(rows, model, stage, score_column, lower_is_better=True):
    """Write results/tuning/<model>_<stage>.csv, one row per combination, worst score first (the best is the LAST row)."""
    table = pd.DataFrame(rows).sort_values(score_column, ascending=not lower_is_better, na_position="first", kind="stable").reset_index(drop=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    path = TUNING_DIR / f"{model}_{stage}.csv"
    table.to_csv(path, index=False)
    return table, path


def validation_scores(split, y_val_pred):
    """Every metric on the validation rows (price in MAD), as val_RMSE, val_MAE, val_R2, val_MAPE."""
    metrics = compute_metrics(split.y_val, y_val_pred, split.inverse_transform_price)
    return {f"val_{name}": value for name, value in rounded(metrics).items()}


def best_params(table, param_names):
    """The parameters of the last (best) row, as plain Python values (ints stay ints) ready to paste into BEST_PARAMS."""
    return table[param_names].iloc[[-1]].to_dict("records")[0]


def grid_search(model, stage, param_grid, evaluate, score_column="val_RMSE", lower_is_better=True):
    """Try every combination in param_grid. evaluate(params) returns a dict of scores that includes score_column.
    The CSV is rewritten after each combination, so stopping early still keeps the results so far."""
    combinations = list(ParameterGrid(param_grid))
    print(f"Grid: {len(combinations)} combinations")
    rows = []
    for number, params in enumerate(combinations, start=1):
        scores, seconds = timed(evaluate, params)
        rows.append({**{name: params[name] for name in param_grid}, **scores, "time_s": round(seconds, 1)})
        table, path = save_tuning_csv(rows, model, stage, score_column, lower_is_better)
        print(f"[{number}/{len(combinations)}] {score_column} {scores[score_column]:,.2f} ({seconds:.0f}s) | best so far {table[score_column].iloc[-1]:,.2f} | {params}")
    return table, path
