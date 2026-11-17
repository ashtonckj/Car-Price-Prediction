"""Tuned LSTM (Part B): trains the patch-based LSTM with BEST_PARAMS and reports 5-fold CV and test results."""
import math
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
    "b1_tuned": {'patch_size': 30, 'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 128},
}

MAX_EPOCHS = 40
PATIENCE = 5     # stop if val loss hasn't improved for this many epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    """Splits the feature vector into patches, runs an LSTM over them, predicts from the last step."""

    def __init__(self, n_features, patch_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.n_steps = math.ceil(n_features / patch_size)
        self.pad = self.n_steps * patch_size - n_features  # zero-padding so the features divide evenly
        self.lstm = nn.LSTM(patch_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = F.pad(x, (0, self.pad)).view(x.shape[0], self.n_steps, self.patch_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


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

    model = LSTMRegressor(X_train.shape[1], params["patch_size"], params["hidden_size"], params["num_layers"], params["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(X_train), device=DEVICE)
        for i in range(0, len(order), params["batch_size"]):
            idx = order[i:i + params["batch_size"]]
            optimizer.zero_grad()
            loss = F.mse_loss(model(X_train[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        sys.exit(f"Add the best parameters for {stage} to BEST_PARAMS at the top of this file (see results/tuning/lstm_{stage}.csv).")
    params = BEST_PARAMS[stage]
    print(f"Device: {DEVICE} | parameters: {params}")
    split = load_data(stage)

    name = "LSTM"
    print(f"\n===== {name} (tuned) =====")
    cv_results = {name: cross_validate(stage, lambda fold: fit_predict(fold, params))}
    results = {name: fit_predict(split, params)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "lstm", "Neural networks", results, split, cv_results)


if __name__ == "__main__":
    main()
