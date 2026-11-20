"""Complexity curves for the tuned models (b1_tuned).
Output: results/b1_tuned/complexity_curve.png and complexity_curve.csv.
"""
import sys
from pathlib import Path

import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared import complexity as cc
from src.shared.core_utils import RANDOM_STATE
from src.shared.stages import load_data
from src.tuned_models import run_dcnv2, run_random_forest

STAGE = "b1_tuned"

# Polynomial Regression and KNN have not been tuned yet. When they are, paste their best parameters here
# (or import them from their src/tuned_models/run_*.py), e.g. POLY_TUNED = {"degree": 2, "fit_intercept": True},
# KNN_TUNED = {"n_neighbors": 10, "weights": "distance", "p": 1}. While a dict is empty that panel uses the default settings.
POLY_TUNED = {}  # "degree" is swept; the other keys go to LinearRegression
KNN_TUNED = {}   # "n_neighbors" is swept; the other keys go to KNeighborsRegressor


def without(params, key):
    return {k: v for k, v in params.items() if k != key}


def label(tuned):
    return "tuned" if tuned else "default, not tuned yet"


def main():
    split = load_data(STAGE)
    n_numeric = cc.n_numeric_columns(STAGE)
    rf_params = run_random_forest.BEST_PARAMS[STAGE]
    dcn_params = run_dcnv2.BEST_PARAMS[STAGE]

    print(f"\nPolynomial Regression (degree; {label(POLY_TUNED)})")
    poly = cc.sweep(split, lambda d: cc.polynomial(d, n_numeric, **without(POLY_TUNED, "degree")), cc.POLY_DEGREES)
    print("\nRandom Forest (max_depth; other settings from run_random_forest.BEST_PARAMS)")
    forest = cc.sweep(split, lambda d: RandomForestRegressor(max_depth=d, **without(rf_params, "max_depth"),
                                                             random_state=RANDOM_STATE, n_jobs=-1), cc.RF_DEPTHS)
    print(f"\nKNN (number of neighbours; {label(KNN_TUNED)})")
    knn = cc.sweep(split, lambda k: KNeighborsRegressor(n_neighbors=k, **without(KNN_TUNED, "n_neighbors"), n_jobs=-1), cc.KNN_K)

    print("\nDCN-V2 (epochs; other settings from run_dcnv2.BEST_PARAMS)")
    torch.manual_seed(RANDOM_STATE)
    model = run_dcnv2.DCNV2Regressor(split.X_train.shape[1], dcn_params["n_cross_layers"], dcn_params["low_rank"], dcn_params["structure"],
                                     dcn_params["deep_layers"], dcn_params["units"], dcn_params["dropout"]).to(run_dcnv2.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=dcn_params["lr"], weight_decay=dcn_params["weight_decay"])
    dcn = cc.epoch_curve(split, model, optimizer, dcn_params["batch_size"], run_dcnv2.predict, flat_target=True, clip_grad=1.0,
                         max_epochs=run_dcnv2.MAX_EPOCHS)

    cc.save(STAGE, [
        ("polynomial_degree", poly, f"Linear family: Polynomial Regression ({label(POLY_TUNED)})",
         "Polynomial degree (simple -> complex)", True, POLY_TUNED.get("degree")),
        ("random_forest_max_depth", forest, "Tree family: Random Forest (tuned)",
         "Maximum tree depth (simple -> complex)", True, rf_params["max_depth"]),
        ("knn_k", knn, f"Distance family: KNN ({label(KNN_TUNED)})",
         "Number of neighbours k (simple -> complex)", True, KNN_TUNED.get("n_neighbors")),
        ("dcnv2_epochs", dcn, "Neural family: DCN-V2 (tuned)",
         "Training epochs (simple -> complex)", False, None),  # the tuned run stops early; its stop is near the green line
    ], f"Model complexity curves, tuned settings ({STAGE}): training vs validation error")


if __name__ == "__main__":
    main()
