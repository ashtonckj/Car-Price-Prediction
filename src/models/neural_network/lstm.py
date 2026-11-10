import sys
import time
from pathlib import Path

from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.neural_network.neural_network_utils import predict, train_torch
from src.shared.core_utils import evaluate, load_data


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, features):  # each feature is treated as one time step
        sequence_output, _ = self.lstm(features.unsqueeze(-1))
        return self.output_layer(sequence_output[:, -1])


X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

t0 = time.time()
model = train_torch(LSTMRegressor(), X_train, y_train, X_val, y_val)
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = predict(model, X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("LSTM", y_test, y_pred, inverse_transform_price)
