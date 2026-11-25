"""Grid search for the neural networks' hyperparameters (Part B)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.experiment import choose_models
from src.shared.stages import choose_stage, load_data
from src.shared.tuning import tune_family, validation_scores
from src.tuned_models.run_neural import DEVICE, predict, train_model

# model name: (CSV name, parameter grid)
PARAM_GRIDS = {
    # 3 x 3 x 2 x 2 x 3 x 2 x 1 = 216 combinations
    "MLP": ("neural_tuning_mlp", {
        "n_layers": [1, 2, 3],
        "units": [128, 256, 512],
        "activation": ["relu", "gelu"],
        "dropout": [0.0, 0.2],
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [256],
    }),
    # 4 x 2 x 2 x 3 x 2 x 3 x 2 x 1 x 1 = 576 combinations
    "DCN-V2": ("neural_tuning_dcnv2", {
        "n_cross_layers": [1, 2, 3, 4],
        "low_rank": [0, 32],                    # 0 = full-rank cross matrix, otherwise low-rank (U V^T)
        "structure": ["stacked", "parallel"],
        "deep_layers": [1, 2, 3],
        "units": [128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-3, 3e-3],
        "weight_decay": [1e-5],
        "batch_size": [256],
    }),
    # 3 x 2 x 2 x 2 x 2 x 1 x 1 = 48 combinations.
    "TabNet": ("neural_tuning_tabnet", {
        "n_d": [8, 16, 32],                 # width of the decision step (and of the attention step)
        "n_steps": [3, 5],                  # number of decision steps
        "gamma": [1.0, 1.5],                # how much a feature may be reused across steps (1.0 = once)
        "lambda_sparse": [1e-4, 1e-3],      # sparsity regularisation of the feature masks
        "lr": [1e-2, 2e-2],
        "batch_size": [1024],
        "virtual_batch_size": [128],
    }),
}


def main():
    stage = choose_stage("B")
    names = choose_models(list(PARAM_GRIDS), "tune")
    print(f"Device: {DEVICE}")
    split = load_data(stage)

    def make_evaluate(name):
        def evaluate(params):
            model, _ = train_model(name, params, split)
            return validation_scores(split, predict(model, split.X_val))
        return evaluate

    tune_family(stage, names, PARAM_GRIDS, make_evaluate, "src/tuned_models/run_neural.py")


if __name__ == "__main__":
    main()
