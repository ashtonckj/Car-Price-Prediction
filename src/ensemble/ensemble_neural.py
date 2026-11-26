"""Ensemble of the tuned PyTorch networks (MLP, DCN-V2): Ridge stacking and optimal weighted blend.
See src/shared/stacking.py. Results: results/b2_ensemble/MLP_DCN_*."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stacking import BASE_STAGE, STAGE, network_learner, stack_and_blend
from src.shared.stages import load_data
from src.tuned_models.run_neural import BEST_PARAMS, train_dcnv2, train_mlp


def main():
    split = load_data(STAGE)
    params = BEST_PARAMS[BASE_STAGE]
    learners = {"MLP": network_learner(train_mlp, params["MLP"], split),
                "DCN-V2": network_learner(train_dcnv2, params["DCN-V2"], split)}
    stack_and_blend(split, learners, "MLP_DCN", "Ensemble (MLP + DCN-V2)")


if __name__ == "__main__":
    main()
