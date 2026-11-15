"""Grid search for the MLP's hyperparameters on a Part B stage: every combination in PARAM_GRID is trained once.

Each combination trains on the 70% train rows with early stopping on the 10% val rows and is scored on the val rows
(the test set is never used). Results: results/tuning/mlp_<stage>.csv, worst to best R2 of log price, so the best
parameters are the LAST row. Copy them into BEST_PARAMS in src/tuned_models/run_mlp.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from src.tuned_models.run_mlp import DEVICE, predict, train_model

# 3 x 3 x 2 x 2 x 3 x 2 x 1 = 216 combinations, about 25 minutes. Every extra value multiplies the count:
# adding batch size 128 makes it 432 and much slower (small batches are about 3x slower per run).
# The default MLP of neural_models.py (2 layers, 128, relu, no dropout, lr 1e-3, no weight decay, batch 256) is one of the rows.
PARAM_GRID = {
    "n_layers": [1, 2, 3],
    "units": [128, 256, 512],  # width of the first layer; later layers halve
    "activation": ["relu", "gelu"],
    "dropout": [0.0, 0.2],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4],
    "batch_size": [256],
}


def main():
    stage = choose_stage("B")
    print(f"Device: {DEVICE}")
    split = load_data(stage)

    def evaluate(params):
        model, _ = train_model(params, split)
        return validation_scores(split, predict(model, split.X_val))

    table, path = grid_search("mlp", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_mlp.py and run it.")


if __name__ == "__main__":
    main()
