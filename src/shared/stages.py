"""The stages the scripts can run. Part A = default parameters (src/models), Part B = tuning (src/tuning, src/tuned_models),
ensemble = Part B step 2 (src/ensemble: the tuned models of a Part B stage combined).
To add a stage: write its load_data_* in core_utils.py and add a line to STAGES."""
import sys

from src.shared.core_utils import load_data_full, load_data_minimal

STAGES = {
    "a1_baseline": {"part": "A", "title": "A1 baseline (minimal preprocessing)", "load": load_data_minimal},
    "a2_preprocessed": {"part": "A", "title": "A2 full preprocessing", "load": load_data_full},
    "b1_tuned": {"part": "B", "title": "B1 tuned (full preprocessing, missing values imputed)", "load": load_data_full},
    # Same data as b1_tuned; the ensemble scripts take their base models' BEST_PARAMS from b1_tuned and save here.
    # Its own part, so choose_stage("B") (tuning / tuned-model menus) does not offer it.
    "b2_ensemble": {"part": "ensemble", "title": "B2 ensemble (tuned b1 models combined)", "load": load_data_full},
}


def choose_stage(part="A"):
    """Ask which stage of Part A or Part B to run, by number (the menu is built from STAGES), and return its key."""
    keys = [key for key, stage in STAGES.items() if stage["part"] == part]
    print(f"Which Part {part} stage do you want to run?")
    for number, key in enumerate(keys, start=1):
        print(f"  {number} = {key:<16} {STAGES[key]['title']}")
    choice = input("Type a number and press Enter: ").strip()
    if not (choice.isdigit() and 1 <= int(choice) <= len(keys)):
        sys.exit(f"'{choice}' is not an option.")
    return keys[int(choice) - 1]


def load_data(stage, fold=None):
    """Load a stage's data: the final split (fold=None) or one cross-validation fold."""
    split = STAGES[stage]["load"](fold)
    if fold is None:
        print(f"[{stage}] Features: {split.X_train.shape[1]} | train {len(split.X_train)}, val {len(split.X_val)}, test {len(split.X_test)}")
    return split
