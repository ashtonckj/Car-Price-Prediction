import sys
import time
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_minimal_data

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_minimal_data()

model = DecisionTreeRegressor(random_state=RANDOM_STATE)  # default parameters
t0 = time.time()
model.fit(X_train, y_train.ravel())
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("Decision Tree", y_test, model.predict(X_test), inverse_transform_price, is_log_target=False)