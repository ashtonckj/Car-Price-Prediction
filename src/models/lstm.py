from common import evaluate, load_data, predict, train_torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, features):  # each feature is treated as one time step
        sequence_output, _ = self.lstm(features.unsqueeze(-1))
        return self.output_layer(sequence_output[:, -1])


X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()
model = train_torch(LSTMRegressor(), X_train, y_train, X_val, y_val)
evaluate("LSTM", y_test, predict(model, X_test), inverse_transform_price)