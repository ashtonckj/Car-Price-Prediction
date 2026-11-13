"""Compares every saved model across stages (no retraining). Run it after the model files; output goes to results/comparison/."""
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RESULTS_ROOT
from src.shared.experiment import load_stage_results
from src.shared.plots import plot_metric_bars, plot_stage_comparison
from src.shared.stages import STAGES

OUT_DIR = RESULTS_ROOT / "comparison"
STAGE_METRICS = ["R2_log", "R2", "MAE", "MAPE"]  # compared across stages
SHOWN_COLUMNS = ["Family", "Model", "R2_log", "CV_R2_log_mean", "CV_R2_log_std", "R2", "MAE", "RMSE", "MAPE", "Fit_time_s"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    tables = {stage: load_stage_results(stage) for stage in STAGES}
    tables = {stage: t for stage, t in tables.items() if not t.empty}
    if not tables:
        sys.exit(f"No results found in {RESULTS_ROOT}/<stage>/. Run the model files first.")

    # One ranked table and bar chart per stage
    for stage, table in tables.items():
        print(f"\n===== {STAGES[stage]['title']} ({stage}) =====")
        print(table[[c for c in SHOWN_COLUMNS if c in table]].to_string(index=False, float_format="{:,.4f}".format))
        table.to_csv(OUT_DIR / f"{stage}_table.csv", index=False)
        plot_metric_bars(table, "R2_log", f"{STAGES[stage]['title']}: R2 of log price (test set)", OUT_DIR / f"{stage}_R2_log.png")

    # Stage vs stage
    if len(tables) < 2:
        print("\nOnly one stage has results so far; run the model files with another stage to compare stages.")
    else:
        everything = pd.concat(tables.values(), ignore_index=True)
        everything["Model"] = everything["Family"] + ": " + everything["Model"]
        for metric in STAGE_METRICS:
            pivot = everything.pivot_table(index="Model", columns="Stage", values=metric)[list(tables)]
            pivot = pivot.sort_values(pivot.columns[-1], ascending=metric not in ("R2", "R2_log"))
            pivot.to_csv(OUT_DIR / f"stages_{metric}.csv")
            plot_stage_comparison(pivot, metric, OUT_DIR / f"stages_{metric}.png")
            if metric == "R2_log":
                print("\n===== R2 of log price on the test set, by stage =====")
                print(pivot.to_string(float_format="{:,.4f}".format))

    print(f"\nSaved tables and charts in {OUT_DIR}")


if __name__ == "__main__":
    main()
