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
from src.shared.stages import load_data

STAGE = "a2_preprocessed"

# DCN-V2 training defaults, copied from train_torch() in neural_models.py
DCN_LR = 1e-3
DCN_BATCH_SIZE = 256


def main():
    split = load_data(STAGE)
    n_numeric = cc.n_numeric_columns(STAGE)

    print("\nPolynomial Regression (degree)")
    poly = cc.sweep(split, lambda d: cc.polynomial(d, n_numeric), cc.POLY_DEGREES)
    print("\nRandom Forest (max_depth; everything else default)")
    forest = cc.sweep(split, lambda d: RandomForestRegressor(max_depth=d, random_state=RANDOM_STATE, n_jobs=-1), cc.RF_DEPTHS)
    print("\nKNN (number of neighbours; everything else default)")
    knn = cc.sweep(split, lambda k: KNeighborsRegressor(n_neighbors=k, n_jobs=-1), cc.KNN_K)

    print("\nDCN-V2 (epochs; everything else default)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_STATE)
    model = DCNV2(n_features=split.X_train.shape[1]).to(device)  # 3 low-rank cross layers, 128-64 deep, dropout 0.1
    dcn = cc.epoch_curve(split, model, torch.optim.Adam(model.parameters(), lr=DCN_LR), DCN_BATCH_SIZE, predict)

    cc.save(STAGE, [
        ("polynomial_degree", poly, "Linear family: Polynomial Regression (default)", "Polynomial degree (simple -> complex)", True, None),
        ("random_forest_max_depth", forest, "Tree family: Random Forest (default)", "Maximum tree depth (simple -> complex)", True, None),
        ("knn_k", knn, "Distance family: KNN (default)", "Number of neighbours k (simple -> complex)", True, None),
        ("dcnv2_epochs", dcn, "Neural family: DCN-V2 (default)", "Training epochs (simple -> complex)", False, None),
    ], f"Model complexity curves, default settings ({STAGE}): training vs validation error")


if __name__ == "__main__":
    main()
