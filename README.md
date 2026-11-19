# Car-Price-Prediction

Predicting used-car prices (MAD) on the Moroccan Used Cars Dataset (MUCars-2024, Mendeley Data) with four model families:
linear, tree-based, distance-based and neural networks.

## Setup
```bash
pip install -r requirements.txt
```

## Workflow
Work through the folders in this order. Every script is run with the Run button of your editor. Instead of command-line
arguments, scripts ask in the terminal which stage you want (type its number).

| Step | Folder | What you do there | Output |
|---|---|---|---|
| 1. Explore | `src/analysis/` | Look at the raw and the preprocessed data | `reports/<stage>/` |
| 2. Preprocess | `src/preprocessing/` | Each script builds one version of the dataset from `data/raw/`. The other scripts build the dataset themselves if it is missing, so you rarely run these by hand | `data/processed/` |
| 3. Part A: default models | `src/models/` | Run a model family. Choose a Part A stage; every model is trained with default parameters, cross-validated and scored on the test set | `results/<stage>/` |
| 4. Part B: tune | `src/tuning/` | Run a tuning script. Choose a Part B stage; it tries every combination of the grid at the top of the script and keeps the one with the lowest RMSE. The slow step, run once | `results/tuning/` |
| 5. Part B: tuned models | `src/tuned_models/` | Copy the best parameters from the tuning CSV into the constant at the top of the script, then run it. Choose a Part B stage | `results/<stage>/` |
| 6. Compare | `src/analysis/` | Run the comparison script: it puts every saved result side by side. Nothing is retrained | `results/comparison/` |

### Reading a tuning result
Each tuning run writes one CSV per model and stage in `results/tuning/`: one row per combination, with the parameters and every
metric (RMSE, MAE, R2, MAPE), sorted from the highest to the lowest RMSE. **The lowest RMSE, the best
combination, is the last row.** Copy its parameters into the constant at the top of the matching script in `src/tuned_models/`;
that script can then be run on its own, without repeating the search. The last few rows are often a statistical tie, so compare
them on R2 too and prefer the simpler or faster one (see `PROJECT_STRUCTURE.md`).

## Stages
A stage is one dataset together with the way it was preprocessed. Results are saved in a folder named after the stage.

| Group | Stage | Meaning |
|---|---|---|
| Part A (default parameters) | `a1_baseline` | Same rows as `a2_preprocessed`, minimal preprocessing (missing values filled with a constant, text one-hot encoded), raw price as target |
| | `a2_preprocessed` | Full preprocessing (cleaning, car-name standardisation, imputation, scaling), log price as target, predictions smeared back to MAD |
| Part B (tuned parameters) | `b1_tuned` | Full preprocessing with missing values imputed |

Part B has its own list, so the same tuning and tuned-model scripts can be run on other ways of preprocessing
(for example dropping rows with missing values) once they are added as stages.

## How the results are measured
- The data is split 70% train, 10% validation (early stopping for the neural networks, and the score used while tuning) and 20% test (scored once at the end).
- Cross-validation is 5-fold on the 70% train part. The saved numbers are the mean and standard deviation over the folds.
- Every stage uses the same rows: fixed price rules (5,000–2,000,000 MAD, placeholder prices such as 999999) applied before the split, then the price scope: only cars between the 1% and 99% price quantiles of the training rows (about 15k–630k MAD), in train, validation and test alike.
- Imputation, encoding, scaling and the price-scope limits are always learned from the training rows only; the smearing factor from the validation rows.
- Every metric is computed on the price in MAD, so all stages are comparable.
- Tuning picks the combination with the lowest RMSE. The result tables and graphs rank and show models by R2.

## Adding things
- **A new stage (another way of preprocessing):** write a loader function for it in `src/shared/` and add one line to the stage list there. Every menu then offers it.
- **A new model to tune:** copy an existing pair from `src/tuning/` and `src/tuned_models/` and change the model and the grid.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full layout and the plan for the rest of Part B.
