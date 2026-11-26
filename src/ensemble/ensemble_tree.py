"""Ensemble of the tuned tree-based models (Random Forest, Gradient Boosting): Ridge stacking and optimal weighted blend.
See src/shared/stacking.py. Results: results/b2_ensemble/RF_GB_*."""
import sys
from functools import partial
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.stacking import BASE_STAGE, STAGE, sklearn_learner, stack_and_blend
from src.shared.stages import load_data
from src.tuned_models.run_tree import BEST_PARAMS, build_model


def main():
    split = load_data(STAGE)
    learners = {name: sklearn_learner(partial(build_model, name, BEST_PARAMS[BASE_STAGE][name]))
                for name in ("Random Forest", "Gradient Boosting")}
    stack_and_blend(split, learners, "RF_GB", "Ensemble (Random Forest + Gradient Boosting)")


if __name__ == "__main__":
    main()
