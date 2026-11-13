"""The stages the model files can run. To add one: write its load_data_* in core_utils.py and add a line to STAGES."""
import sys

from src.shared.core_utils import load_data_full, load_data_minimal

STAGES = {
    "a1_baseline": {"title": "A1 baseline (minimal preprocessing)", "load": load_data_minimal},
    "a2_preprocessed": {"title": "A2 full preprocessing", "load": load_data_full},
}


def choose_stage():
    """Ask which stage to run by number (the menu is built from STAGES) and return its key."""
    keys = list(STAGES)
    print("Which stage do you want to run?")
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
