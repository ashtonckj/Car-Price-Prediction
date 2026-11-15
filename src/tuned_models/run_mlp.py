"""Tuned MLP (Part B): trains the MLP with BEST_PARAMS and reports 5-fold CV and test results.
Choose a Part B stage when asked. Results go to results/<stage>/."""
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

# Best parameters per Part B stage. After tuning, copy them from the LAST row of results/tuning/mlp_<stage>.csv.
BEST_PARAMS = {
    # best of the earlier random search; replace with the grid search's best row
    "b1_tuned": {'n_layers': 3, 'units': 512, 'activation': 'gelu', 'dropout': 0.2, 'lr': 0.0003, 'weight_decay': 0.0001, 'batch_size': 256},
}

MAX_EPOCHS = 100  # same limits as the default MLP in neural_models.py, so only the parameters differ
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def build_mlp(n_inputs, params):
    """Hidden layers halve in width: units, units/2, ... (units=128, n_layers=2 is the default 128-64 MLP)."""
    layers, width = [], n_inputs
    for i in range(params["n_layers"]):
        size = max(16, params["units"] // 2**i)
        layers += [nn.Linear(width, size), ACTIVATIONS[params["activation"]](), nn.Dropout(params["dropout"])]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, 1))


def predict(model, X, batch_size=2048):
    model.eval()
    X = torch.as_tensor(X, device=DEVICE)
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]).cpu().numpy()


def train_model(params, split):
    """Train with early stopping on split.X_val. Returns (model with the best weights, best val MSE)."""
    torch.manual_seed(RANDOM_STATE)  # before building the model, so the starting weights are the same every run
    X_train = torch.as_tensor(split.X_train, device=DEVICE)
    y_train = torch.as_tensor(split.y_train, device=DEVICE)
    model = build_mlp(X_train.shape[1], params).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(X_train), device=DEVICE)
        for i in range(0, len(order), params["batch_size"]):
            batch = order[i:i + params["batch_size"]]
            optimizer.zero_grad()
            F.mse_loss(model(X_train[batch]), y_train[batch]).backward()
            optimizer.step()

        val_loss = float(np.mean((predict(model, split.X_val) - split.y_val) ** 2))
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
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def main():
    stage = choose_stage("B")
    if stage not in BEST_PARAMS:
        sys.exit(f"Add the best parameters for {stage} to BEST_PARAMS at the top of this file (see results/tuning/mlp_{stage}.csv).")
    params = BEST_PARAMS[stage]
    print(f"Device: {DEVICE} | parameters: {params}")
    split = load_data(stage)

    name = "MLP"  # same family and model name as the default run, so comparison.py lines the stages up
    print(f"\n===== {name} (tuned) =====")
    cv_results = {name: cross_validate(stage, lambda fold: fit_predict(fold, params))}
    results = {name: fit_predict(split, params)}
    print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
    evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "mlp", "Neural networks", results, split, cv_results)


if __name__ == "__main__":
    main()
