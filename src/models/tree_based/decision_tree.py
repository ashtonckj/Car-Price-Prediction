import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_data

data = load_data()

# Cross-validation
for i, fold in enumerate(data.folds):
    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model.fit(fold.X_train, fold.y_train.ravel())
    y_val_pred = model.predict(fold.X_val)
    val_rmse = np.sqrt(mean_squared_error(fold.y_val, y_val_pred))
    print(f"Fold {i+1}/{data.n_folds} - Val RMSE: {val_rmse:.4f}")

model = DecisionTreeRegressor(random_state=RANDOM_STATE)
t0 = time.time()
model.fit(data.X_train, data.y_train.ravel())
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(data.X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("Decision Tree", data.y_test, y_pred, data.inverse_transform_price, is_log_target=True)
