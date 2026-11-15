"""Grid search for the LSTM's hyperparameters on a Part B stage: every combination in PARAM_GRID is trained once.

Each combination trains on the 70% train rows with early stopping on the 10% val rows and is scored on the val rows
(the test set is never used). Results: results/tuning/lstm_<stage>.csv, worst to best R2 of log price, so the best
parameters are the LAST row. Copy them into BEST_PARAMS in src/tuned_models/run_lstm.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from src.tuned_models.run_lstm import DEVICE, predict, train_model

# 3 x 2 x 2 x 2 x 2 x 1 x 2 = 96 combinations, about 1 hour (a run takes 20-60 s; small batches and small patches are slowest).
# Every extra value multiplies the count. The values are a subset of the lists of the earlier random search.
PARAM_GRID = {
    "patch_size": [15, 30, 45],  # features per time step
    "hidden_size": [64, 256],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.3],
    "lr": [1e-3, 3e-3],
    "weight_decay": [1e-5],
    "batch_size": [128, 256],
}


def main():
    stage = choose_stage("B")
    print(f"Device: {DEVICE}")
    split = load_data(stage)

    def evaluate(params):
        model, _ = train_model(params, split)
        return validation_scores(split, predict(model, split.X_val))

    table, path = grid_search("lstm", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_lstm.py and run it.")


if __name__ == "__main__":
    main()
