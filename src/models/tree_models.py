"""Tree-based models: Decision Tree, Random Forest, Gradient Boosting. Run it and type a stage number."""
import sys
from functools import partial
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE
from src.shared.experiment import fit_predict, run_family
from src.shared.stages import choose_stage, load_data

# All models use default parameters
MODELS = {
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),  # n_jobs only changes speed
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def main():
    stage = choose_stage()
    split = load_data(stage)
    run_family(stage, split, "tree", "Tree-based models", {name: partial(fit_predict, make) for name, make in MODELS.items()})


if __name__ == "__main__":
    main()
