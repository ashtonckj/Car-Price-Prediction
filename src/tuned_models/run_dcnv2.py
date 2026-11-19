"""Tuned DCN V2 (Part B): trains the Deep & Cross Network v2 with BEST_PARAMS and reports 5-fold CV and test results."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data

BEST_PARAMS = {
    "b1_tuned": {'n_cross_layers': 1, 'low_rank': 32, 'structure': 'stacked', 'deep_layers': 3, 'units': 128, 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 1e-05, 'batch_size': 256},
}

MAX_EPOCHS = 100
PATIENCE = 10    # stop if val loss hasn't improved for this many epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossLayer(nn.Module):
    """DCN V2 cross layer: x_{l+1} = x0 * (W x_l + b) + x_l.
    With rank > 0, W is factorised as U V^T (low-rank cross); rank = 0 uses the full d x d matrix."""

    def __init__(self, dim, rank=0):
        super().__init__()
        if rank and rank < dim:
            self.proj = nn.Sequential(nn.Linear(dim, rank, bias=False), nn.Linear(rank, dim))
        else:
            self.proj = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.proj(x) + x


class DCNV2Regressor(nn.Module):
    """Cross network (explicit feature crosses) + deep network (implicit interactions).
    structure='stacked':  x -> cross -> deep -> head
    structure='parallel': x -> [cross | deep] -> concat -> head"""

    def __init__(self, n_features, n_cross_layers, low_rank, structure, deep_layers, units, dropout):
        super().__init__()
        self.structure = structure
        self.cross = nn.ModuleList([CrossLayer(n_features, low_rank) for _ in range(n_cross_layers)])

        # deep tower: widths halve like the MLP (units, units/2, ...)
        deep_in = n_features
        layers, width = [], deep_in
        for i in range(deep_layers):
            size = max(16, units // 2**i)
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.deep = nn.Sequential(*layers)
        deep_out = width if deep_layers > 0 else deep_in

        head_in = deep_out if structure == "stacked" else n_features + deep_out
        self.head = nn.Linear(head_in, 1)

    def forward(self, x):
        x_cross = x
        for layer in self.cross:
            x_cross = layer(x, x_cross)
        if self.structure == "stacked":
            return self.head(self.deep(x_cross)).squeeze(-1)
        return self.head(torch.cat([x_cross, self.deep(x)], dim=1)).squeeze(-1)


def predict(model, X, batch_size=2048):
    model.eval()
    X = torch.as_tensor(X, device=DEVICE)
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]).cpu().numpy()


def train_model(params, split):
    """Train with early stopping on split.X_val. Returns (model with the best weights, best val MSE)."""
    torch.manual_seed(RANDOM_STATE)
    X_train = torch.as_tensor(split.X_train, device=DEVICE)
    y_train = torch.as_tensor(split.y_train.ravel(), device=DEVICE)
    y_val = split.y_val.ravel()

    model = DCNV2Regressor(X_train.shape[1], params["n_cross_layers"], params["low_rank"], params["structure"],
                           params["deep_layers"], params["units"], params["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(X_train), device=DEVICE)
        for i in range(0, len(order), params["batch_size"]):
            idx = order[i:i + params["batch_size"]]
            optimizer.zero_grad()
            loss = F.mse_loss(model(X_train[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # cross layers multiply features, so keep gradients bounded
            optimizer.step()

        val_loss = float(np.mean((predict(model, split.X_val) - y_val) ** 2))
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_loss


def fit_predict(split, params):
    (model, _), fit_time = timed(train_model, params, split)
    y_pred, predict_time = timed(predict, model, split.X_test)
    y_pred = split.smear(predict(model, split.X_val), y_pred)  # log-price bias correction
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    if stage not in BEST_PARAMS:
        sys.exit(f"Add the best parameters for {stage} to BEST_PARAMS at the top of this file (see results/tuning/dcnv2_{stage}.csv).")
    params = BEST_PARAMS[stage]
    print(f"Device: {DEVICE} | parameters: {params}")
    split = load_data(stage)

    name = "DCN-V2"
    print(f"\n===== {name} (tuned) =====")
    cv_results = {name: cross_validate(stage, lambda fold: fit_predict(fold, params))}
    results = {name: fit_predict(split, params)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "dcnv2", "Neural networks", results, split, cv_results)


if __name__ == "__main__":
    main()