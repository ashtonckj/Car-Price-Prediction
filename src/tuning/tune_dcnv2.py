"""Grid search for the DCN V2's hyperparameters on a Part B stage: every combination in PARAM_GRID is trained once."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stages import choose_stage, load_data
from src.shared.tuning import best_params, grid_search, validation_scores
from src.tuned_models.run_dcnv2 import DEVICE, predict, train_model

# 4 x 2 x 2 x 3 x 2 x 2 x 3 x 1 x 1 = 576 combinations
PARAM_GRID = {
    "n_cross_layers": [1, 2, 3, 4],
    "low_rank": [0, 32],                    # 0 = full-rank cross matrix, otherwise low-rank (U V^T)
    "structure": ["stacked", "parallel"],
    "deep_layers": [1, 2, 3],
    "units": [128, 256],
    "dropout": [0.0, 0.2],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5],
    "batch_size": [256],
}


def main():
    stage = choose_stage("B")
    print(f"Device: {DEVICE}")
    split = load_data(stage)

    def evaluate(params):
        model, _ = train_model(params, split)
        return validation_scores(split, predict(model, split.X_val))

    table, path = grid_search("dcnv2", stage, PARAM_GRID, evaluate)
    print(f"\nBest parameters: {best_params(table, list(PARAM_GRID))}")
    print(f"Saved {path} (best = last row). Next: put them in BEST_PARAMS in src/tuned_models/run_dcnv2.py and run it.")


if __name__ == "__main__":
    main()