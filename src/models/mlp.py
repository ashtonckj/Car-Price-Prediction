from common import evaluate, load_data, predict, train_torch
from torch import nn

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()
number_of_features = X_train.shape[1]

model = nn.Sequential(
    nn.Linear(number_of_features, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
model = train_torch(model, X_train, y_train, X_val, y_val)
evaluate("MLP", y_test, predict(model, X_test), inverse_transform_price)