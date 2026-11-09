import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

DATA_PATH = "data/processed/cars.csv"
RANDOM_STATE = 42   
TEST_SIZE = 0.2     # 80% train / 20% test
CATEGORICAL_COLS = ["Brand", "Model"]

# ------------------------- Load and split data -----------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Price"])

y = np.log1p(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

encoder = ColumnTransformer(
    [("target_enc", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE),
      CATEGORICAL_COLS)],
    remainder="passthrough",
)

pipeline = Pipeline([
    ("encode", encoder),
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor()),   # all default parameters
])

# ----------------------------- Train ---------------------------------
t0 = time.time()
pipeline.fit(X_train, y_train)
fit_time = time.time() - t0

# ---------------------------- Predict --------------------------------
t0 = time.time()
y_pred_log = pipeline.predict(X_test)
predict_time = time.time() - t0

# ---------------------------- Evaluate -------------------------------
y_true = np.expm1(y_test)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)   # prices cannot be negative

print("\n===== KNN Regressor (default parameters) =====")
print(f"R2 (original price)  : {r2_score(y_true, y_pred):.4f}")
print(f"RMSE                 : {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
print(f"MAE                  : {mean_absolute_error(y_true, y_pred):,.2f}")
print(f"MAPE (%)             : {100 * mean_absolute_percentage_error(y_true, y_pred):.2f}")
print(f"R2 (log scale)       : {r2_score(y_test, y_pred_log):.4f}")
print(f"Fit time (s)         : {fit_time:.2f}")
print(f"Predict time (s)     : {predict_time:.2f}")