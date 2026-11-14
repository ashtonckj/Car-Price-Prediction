# Car-Price-Prediction

Predicting used-car prices (MAD) on the Moroccan Used Cars Dataset (MUCars-2024, Mendeley Data)
with four model families: linear, tree-based, distance-based and neural networks.

## Setup
```bash
pip install -r requirements.txt
```

## How to run (in order)
Every file can be run with the Run button in your editor; nothing needs command-line arguments.

| Step | File | What it does | Output |
|---|---|---|---|
| EDA | `src/analysis/a1_baseline_eda.py` | Explores the raw data | `reports/a1_baseline/` |
| | `src/analysis/a2_preprocessed_eda.py` | Correlation heatmap of the fully preprocessed data | `reports/a2_preprocessed/` |
| Preprocessing | `src/preprocessing/minimal.py` | Minimal: drop rows without price, missing numbers -> 0, drop rows with missing text | `data/processed/cars_minimal.csv` |
| | `src/preprocessing/full.py` | Full: cleaning, numeric conversion, feature selection, log price | `data/processed/cars_full.csv` |
| Models | `src/models/linear_models.py`, `tree_models.py`, `distance_models.py`, `neural_models.py` | Asks for a stage number, then runs 5-fold CV and a final test-set score for each model | `results/<stage>/<family>_results.json` and `_pred_vs_actual.png` |
| Comparison | `src/models/comparison.py` | Ranks all saved models per stage and compares stages (no retraining) | `results/comparison/` |

The model files create the preprocessed CSV themselves if it is missing.
Set `RUN_CV = False` at the top of a model file to skip cross-validation (much faster, test-set results only).

## Stages
| Number | Stage | Preprocessing | Target |
|---|---|---|---|
| 1 | `a1_baseline` | minimal; one-hot encoding of every text column, no scaling | Price |
| 2 | `a2_preprocessed` | full; group-mode imputation, scaling | Log Price |

All metrics (RMSE, MAE, R2, MAPE and R2 of log price) are computed on the price in MAD, so the stages are comparable.

## How the data is split
20% test (scored once at the end), 10% validation (early stopping for the neural networks) and 70% train.
The 5-fold cross-validation runs on the 70% train part, and the saved CV numbers are the mean and std over the 5 folds.
Imputation, encoding and scaling are fitted on the training rows only.

## Adding a stage
Write a `load_data_<name>(fold=None)` function in `src/shared/core_utils.py` and add one line to `STAGES` in `src/shared/stages.py`;
every model file then offers it in its menu.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the layout and the plan for Part B (tuning, ensembles, real-world data).
