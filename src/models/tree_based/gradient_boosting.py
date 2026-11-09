import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_data

X_train, y_train, X_val, y_val, X_test, y_test, inverse_transform_price = load_data()

model = GradientBoostingRegressor(random_state=RANDOM_STATE)  # default parameters
model.fit(X_train, y_train.ravel())
evaluate("Gradient Boosting", y_test, model.predict(X_test), inverse_transform_price)