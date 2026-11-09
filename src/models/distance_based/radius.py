import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import evaluate, load_data

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = RadiusNeighborsRegressor()  # default parameters (radius=1.0)

t0 = time.time()
model.fit(X_train, y_train.ravel())
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

# Cars with no neighbour inside the radius come back as NaN
empty = np.isnan(y_pred)
y_pred[empty] = y_train.mean()  # fallback: average training target
print(f"Empty neighbourhoods: {100 * empty.mean():.2f}% of test cars (filled with train mean)")

evaluate("Radius Neighbours", y_test, y_pred, inverse_transform_price)