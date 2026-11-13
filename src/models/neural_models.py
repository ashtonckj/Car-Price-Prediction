"""Neural networks: MLP, LSTM, TabNet. Run it and type a stage number. The 10% val set is used for early stopping."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate
from src.shared.experiment import cross_validate, save_results, timed
from src.shared.stages import choose_stage, load_data


# Training utilities (shared by the PyTorch models)
def predict(model, X, batch_size=512):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(X[i:i + batch_size]).to(device)).cpu().numpy() for i in range(0, len(X), batch_size)])


def train_torch(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=1e-3, batch_size=256, patience=10):
    torch.manual_seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer, loss_function = (
        torch.optim.Adam(model.parameters(), lr=learning_rate),
        nn.MSELoss(),
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss_function(model(X_batch), y_batch).backward()
            optimizer.step()
        val_loss = float(np.mean((predict(model, X_val) - y_val) ** 2))
        print(f"epoch {epoch + 1} | val_loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model


class RNNRegressor(nn.Module):
    """GRU or LSTM; each feature is treated as one time step."""

    def __init__(self, rnn_class, hidden_size=64):
        super().__init__()
        self.rnn = rnn_class(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, features):
        sequence_output, _ = self.rnn(features.unsqueeze(-1))
        return self.output_layer(sequence_output[:, -1])


# Models: each run_*(split) trains on split.X_train (early stopping on split.X_val) and predicts split.X_test
def fit_and_predict(network, split):
    model, fit_time = timed(train_torch, network, split.X_train, split.y_train, split.X_val, split.y_val)
    y_pred, predict_time = timed(predict, model, split.X_test)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


def run_mlp(split):
    network = nn.Sequential(
        nn.Linear(split.X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    return fit_and_predict(network, split)


def run_lstm(split):
    return fit_and_predict(RNNRegressor(nn.LSTM), split)


def run_tabnet(split):
    from pytorch_tabnet.tab_model import (
        TabNetRegressor,  # imported here so the other models still work if it is missing
    )

    model = TabNetRegressor(seed=RANDOM_STATE, verbose=10)
    _, fit_time = timed(
        model.fit,
        split.X_train,
        split.y_train,
        eval_set=[(split.X_val, split.y_val)],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
    )
    y_pred, predict_time = timed(model.predict, split.X_test)
    return {"model": model, "y_pred": y_pred, "fit_time": fit_time, "predict_time": predict_time}


MODELS = {"MLP": run_mlp, "LSTM": run_lstm, "TabNet": run_tabnet}


def main():
    stage = choose_stage()
    split = load_data(stage)

    results, cv_results = {}, {}
    for name, run in MODELS.items():
        print(f"\n===== {name} =====")
        cv_results[name] = cross_validate(stage, run)
        results[name] = run(split)
        print(f"Fit time (s): {results[name]['fit_time']:.2f} | Predict time (s): {results[name]['predict_time']:.2f}")
        evaluate(name, split.y_test, results[name]["y_pred"], split.inverse_transform_price)

    save_results(stage, "neural", "Neural networks", results, split, cv_results)


if __name__ == "__main__":
    main()
