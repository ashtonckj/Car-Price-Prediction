"""Tune an LSTM on a2_preprocessed (random search on the validation set), then evaluate the best config.

Save as src/models/tune_lstm.py (or wherever your model files live; ROOT_DIR assumes src/<folder>/file.py).
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, RESULTS_ROOT, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import load_data

STAGE = "a2_preprocessed"
N_TRIALS = 20      # number of random configurations to try
MAX_EPOCHS = 40    # upper limit per training run
PATIENCE = 5       # stop if val loss hasn't improved for this many epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEARCH_SPACE = {
    "patch_size": [10, 15, 25, 30, 45],   # features per time step
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [128, 256, 512],
}


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


def tune(split):
    """Random search: returns (best params, list of all trials)."""
    rng = random.Random(RANDOM_STATE)
    trials = []
    for t in range(N_TRIALS):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        (_, val_loss), seconds = timed(train_model, params, split)
        trials.append({"params": params, "val_mse": round(val_loss, 5), "time_s": round(seconds, 1)})
        print(f"Trial {t + 1}/{N_TRIALS}: val MSE {val_loss:.4f} ({seconds:.0f}s) {params}")
    best = min(trials, key=lambda tr: tr["val_mse"])
    return best["params"], trials


def main():
    print(f"Device: {DEVICE}")
    split = load_data(STAGE)

    best_params, trials = tune(split)
    print(f"\nBest params: {best_params}")

    out_dir = RESULTS_ROOT / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lstm_tuning.json").write_text(json.dumps({"best_params": best_params, "trials": trials}, indent=2))

    def fit_predict(s):
        (model, _), fit_time = timed(train_model, best_params, s)
        y_pred, predict_time = timed(predict, model, s.X_test)
        return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}

    name = "LSTM (tuned)"
    print(f"\n===== {name} =====")
    cv_results = {name: cross_validate(STAGE, fit_predict)}
    results = {name: fit_predict(split)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(STAGE, "lstm", "LSTM", results, split, cv_results)


if __name__ == "__main__":
    main()