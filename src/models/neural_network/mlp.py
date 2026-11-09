import sys
from pathlib import Path

from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.neural_network.neural_network_utils import predict, train_torch
from src.shared.core_utils import evaluate, load_data

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
model = train_torch(model, X_train, y_train, X_val, y_val)
evaluate("MLP", y_test, predict(model, X_test), inverse_transform_price)