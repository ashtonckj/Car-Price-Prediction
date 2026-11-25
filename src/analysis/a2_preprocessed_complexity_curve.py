"""Complexity curves for the default models (a2_preprocessed).
Output: results/a2_preprocessed/complexity_curve.png and complexity_curve.csv.
"""
import sys
from pathlib import Path

import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.neural_models import DCNV2, predict
from src.shared import complexity as cc
from src.shared.core_utils import RANDOM_STATE

STAGE = "a2_preprocessed"

# DCN-V2 training defaults, copied from train_torch() in neural_models.py
DCN_LR = 1e-3
DCN_BATCH_SIZE = 256
DCN_PATIENCE = 10  # early stopping, as in train_torch(patience=10)

# Tree depth range for the validation zoom.
# Exclude early underfitting (depths 10, 20) to zoom into the plateau/minimum region.
# Examples: (30, 70), (30, None), or a list like ["30", "40", "50", "60", "70"]
RF_ZOOM_RANGE = (40, 100)


def poly_curve(split):
    n_numeric = cc.n_numeric_columns(STAGE)
    return cc.sweep(split, lambda d: cc.polynomial(d, n_numeric), cc.POLY_DEGREES)


def forest_curve(split):
    return cc.sweep(split, lambda d: RandomForestRegressor(max_depth=d, random_state=RANDOM_STATE, n_jobs=-1), cc.RF_DEPTHS)


def knn_curve(split):
    return cc.sweep(split, lambda k: KNeighborsRegressor(n_neighbors=k, n_jobs=-1), cc.KNN_K)


def dcn_curve(split):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_STATE)
    model = DCNV2(n_features=split.X_train.shape[1]).to(device)  # 3 low-rank cross layers, 128-64 deep, dropout 0.1
    return cc.epoch_curve(split, model, torch.optim.Adam(model.parameters(), lr=DCN_LR), DCN_BATCH_SIZE, predict, patience=DCN_PATIENCE)


CURVES = {
    "polynomial_degree": cc.Curve("Polynomial Regression (degree)", "Linear family: Polynomial Regression (default)", "Polynomial degree (simple -> complex)", True, None, poly_curve),
    "random_forest_max_depth": cc.Curve("Random Forest (max_depth)", "Tree family: Random Forest (default)", "Maximum tree depth (simple -> complex)", True, None, forest_curve, RF_ZOOM_RANGE),
    "knn_k": cc.Curve("KNN (number of neighbours)", "Distance family: KNN (default)", "Number of neighbours k (simple -> complex)", True, None, knn_curve),
    "dcnv2_epochs": cc.Curve("DCN-V2 (epochs, slow)", "Neural family: DCN-V2 (default)", "Training epochs (simple -> complex)", False, None, dcn_curve),
}


def main():
    cc.run(STAGE, CURVES, f"Model complexity curves, default settings ({STAGE}): training vs validation error")


if __name__ == "__main__":
    main()
