"""Complexity curves for the tuned models (b1_tuned).
Output: results/b1_tuned/complexity_curve.png and complexity_curve.csv.
"""
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared import complexity as cc
from src.shared.core_utils import RANDOM_STATE
from src.tuned_models import run_distance, run_linear, run_neural, run_tree

STAGE = "b1_tuned"

RF_ZOOM_RANGE = (40, 100)

POLY_PARAMS = run_linear.BEST_PARAMS[STAGE]["Polynomial Regression"]
RF_PARAMS = run_tree.BEST_PARAMS[STAGE]["Random Forest"]
KNN_PARAMS = run_distance.BEST_PARAMS[STAGE]["KNN"]
DCN_PARAMS = run_neural.BEST_PARAMS[STAGE]["DCN-V2"]


def poly_curve(split):
    n_numeric = cc.n_numeric_columns(STAGE)
    return cc.sweep(split, lambda d: run_linear.build_model("Polynomial Regression", {**POLY_PARAMS, "degree": d}, n_numeric), cc.POLY_DEGREES)


def forest_curve(split):
    return cc.sweep(split, lambda d: run_tree.build_model("Random Forest", {**RF_PARAMS, "max_depth": d}), cc.RF_DEPTHS)


def knn_curve(split):
    return cc.sweep(split, lambda k: run_distance.build_model("KNN", {**KNN_PARAMS, "n_neighbors": k}), cc.KNN_K)


def dcn_curve(split):
    p = DCN_PARAMS
    torch.manual_seed(RANDOM_STATE)
    model = run_neural.DCNV2Regressor(split.X_train.shape[1], p["n_cross_layers"], p["low_rank"], p["structure"],
                                     p["deep_layers"], p["units"], p["dropout"]).to(run_neural.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    return cc.epoch_curve(split, model, optimizer, p["batch_size"], run_neural.predict, flat_target=True, clip_grad=1.0,
                          max_epochs=run_neural.MAX_EPOCHS)

CURVES = {
    "polynomial_degree": cc.Curve("Polynomial Regression (degree)", "Linear family: Polynomial Regression (tuned)", "Polynomial degree (simple -> complex)", True, POLY_PARAMS["degree"], poly_curve),
    "random_forest_max_depth": cc.Curve("Random Forest (max_depth, slow)", "Tree family: Random Forest (tuned)", "Maximum tree depth (simple -> complex)", True, RF_PARAMS["max_depth"], forest_curve, RF_ZOOM_RANGE),
    "knn_k": cc.Curve("KNN (number of neighbours)", "Distance family: KNN (tuned)", "Number of neighbours k (simple -> complex)", True, KNN_PARAMS["n_neighbors"], knn_curve),
    "dcnv2_epochs": cc.Curve("DCN-V2 (epochs, slow)", "Neural family: DCN-V2 (tuned)", "Training epochs (simple -> complex)", False, None, dcn_curve),
}


def main():
    cc.run(STAGE, CURVES, f"Model complexity curves, tuned settings ({STAGE}): training vs validation error")


if __name__ == "__main__":
    main()