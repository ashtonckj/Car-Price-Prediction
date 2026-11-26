"""Ensemble of four tuned models from different families (Polynomial Regression, Gradient Boosting, Kernel Regression,
MLP): Ridge stacking and optimal weighted blend. See src/shared/stacking.py. Results: results/b2_ensemble/PR_GB_KR_MLP_*."""
import sys
from functools import partial
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import n_numeric_columns
from src.shared.stacking import (
    BASE_STAGE,
    STAGE,
    network_learner,
    sklearn_learner,
    stack_and_blend,
)
from src.shared.stages import load_data
from src.tuned_models import run_distance, run_linear, run_neural, run_tree


def main():
    split = load_data(STAGE)
    linear, tree, distance = (module.BEST_PARAMS[BASE_STAGE] for module in (run_linear, run_tree, run_distance))
    learners = {
        "Polynomial Regression": sklearn_learner(partial(run_linear.build_model, "Polynomial Regression",
                                                         linear["Polynomial Regression"], n_numeric_columns(split))),
        "Gradient Boosting": sklearn_learner(partial(run_tree.build_model, "Gradient Boosting", tree["Gradient Boosting"])),
        "Kernel Regression": sklearn_learner(partial(run_distance.build_model, "Kernel Regression", distance["Kernel Regression"])),
        "MLP": network_learner(run_neural.train_mlp, run_neural.BEST_PARAMS[BASE_STAGE]["MLP"], split),
    }
    stack_and_blend(split, learners, "PR_GB_KR_MLP", "Ensemble (Polynomial + GB + Kernel Regression + MLP)")


if __name__ == "__main__":
    main()
