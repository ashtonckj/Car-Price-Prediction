# Car Price Prediction

Predicting used-car prices (in Moroccan dirham, MAD) from the **Moroccan Used Cars Dataset** (MUCars-2024, Mendeley Data).
Four model families are compared — **linear, tree-based, distance-based and neural networks** — first with default
parameters (Part A), then tuned, combined into ensembles and tested on real-world European listings (Part B).

The reasoning and numbers behind every decision are in [REPORT.md](REPORT.md).

## Setup

Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
git lfs pull                      # large files (saved models, raw European data) are stored in Git LFS
```

For an NVIDIA GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org) instead of the default one.

## How to run

Every script runs with your editor's Run button (or `python src/<folder>/<file>.py`). Instead of arguments, it asks in the
terminal which **stage** to use and which **models** to run (numbers separated by commas, Enter = all).

1. **Explore** — `src/analysis/a1_baseline_eda.py`, `a2_preprocessed_eda.py`
2. **Default models (Part A)** — `src/models/<family>_models.py`
3. **Tune (Part B)** — `src/tuning/tune_<family>.py`, then paste the best parameters into `BEST_PARAMS` in
   `src/tuned_models/run_<family>.py` and run it
4. **Ensembles** — `src/ensemble/*.py`
5. **Compare and check** — `src/analysis/comparison.py`, the complexity-curve scripts, `shap_analysis.py`
6. **New data** — `src/external/evaluate_saved_models.py`

The processed datasets are rebuilt automatically when missing, so the preprocessing scripts rarely need running by hand.

## Project structure

```
Car-Price-Prediction/
├── data/
│   ├── raw/cars_dataframe.csv            original Moroccan dataset
│   ├── external/                         European listings (Hugging Face KTHESAT, CC BY-NC 4.0)
│   └── processed/                        cleaned datasets built by src/preprocessing/
├── src/
│   ├── preprocessing/                    raw data -> data/processed/
│   │   ├── minimal.py                    a1: same rows as full.py, missing values filled with a constant
│   │   ├── full.py                       a2/b1: cleaning, price rules, car-name standardisation, log price
│   │   └── external_europe.py            European listings in the same columns as cars_full.csv
│   ├── models/                           Part A: one file per family, default parameters
│   ├── tuning/                           Part B: grid search per family -> results/<stage>/<family>_tuning_<model>.csv
│   ├── tuned_models/                     Part B: each family with its BEST_PARAMS (also saves the models)
│   ├── ensemble/                         Part B: stacking and blending of tuned models
│   ├── external/                         evaluate_saved_models.py: saved models tested on the European listings
│   ├── analysis/                         EDA, stage comparison, complexity curves, SHAP
│   └── shared/                           code used by every script
│       ├── stages.py                     list of stages and the stage menu
│       ├── core_utils.py                 data loaders, 70/10/20 split, CV folds, metrics
│       ├── experiment.py                 cross-validation, run loop, saving results
│       ├── tuning.py                     grid search (resumes a stopped search)
│       ├── stacking.py                   shared ensemble code
│       ├── complexity.py                 shared complexity-curve code
│       ├── saved_models.py               save / load models with their preprocessing
│       └── plots.py                      shared graphs
├── results/                              one folder per stage: metrics (JSON / CSV) and graphs
├── reports/                              one folder per stage: EDA figures
├── saved_models/                         one folder per stage: trained models with their preprocessing (.joblib)
├── requirements.txt
├── README.md
└── REPORT.md
```

## Stages

A stage is one way of preparing the data; results are saved in a folder with its name.

| Stage | Meaning |
|---|---|
| `a1_baseline` | Minimal preprocessing, raw price as target, default parameters |
| `a2_preprocessed` | Full preprocessing, log price as target, default parameters |
| `b1_tuned` | Same data as a2, tuned parameters |
| `b2_ensemble` | Tuned b1 models combined (stacking and blending) |
| `b3_external` | Saved a2/b1 models tested on European listings |

Data split: 70% train, 10% validation, 20% test, plus 5-fold cross-validation on the training part. Everything learned
from the data (imputation, encoding, scaling, price range) comes from the training rows only, and every metric is computed
on the price in MAD.
