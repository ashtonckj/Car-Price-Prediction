import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.shared.core_utils import RANDOM_STATE


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
