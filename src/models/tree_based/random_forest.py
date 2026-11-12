import sys
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import shap
import pandas as pd


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import RANDOM_STATE, evaluate, load_data

data = load_data()

# Cross-validation
for i, fold in enumerate(data.folds):
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(fold.X_train, fold.y_train.ravel())
    y_val_pred = model.predict(fold.X_val)
    val_rmse = np.sqrt(mean_squared_error(fold.y_val, y_val_pred))
    print(f"Fold {i+1}/{data.n_folds} - Val RMSE: {val_rmse:.4f}")

model = RandomForestRegressor(random_state=RANDOM_STATE)
t0 = time.time()
model.fit(data.X_train, data.y_train.ravel())
print(f"Fit time (s): {time.time() - t0:.2f}")

t0 = time.time()
y_pred = model.predict(data.X_test)
print(f"Predict time (s): {time.time() - t0:.2f}")

evaluate("Random Forest", data.y_test, y_pred, data.inverse_transform_price, is_log_target=True)

# 1. Subsample test set (500 rows keeps unpruned RF SHAP computation under ~20 mins)
rng = np.random.default_rng(RANDOM_STATE)
sample_size = min(500, len(data.X_test))
sample_idx = rng.choice(len(data.X_test), size=sample_size, replace=False)
X_test_sample = data.X_test[sample_idx]


# 2. Compute SHAP values (or load from disk if already computed)
shap_cache_path = Path("rf_shap_values.joblib")
if shap_cache_path.exists():
    shap_values = joblib.load(shap_cache_path)
    print("Loaded cached SHAP values from disk.")
else:
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample, check_additivity=False)
    joblib.dump(shap_values, shap_cache_path)
    print(f"SHAP computation time ({sample_size} samples): {time.time() - t0:.2f}s")


# Attach real feature names (works even on reloaded cache)
shap_values.feature_names = data.feature_names


# 3. Save ranked Mean |SHAP| table to CSV (useful for report tables & Part B feature selection)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": data.feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv("rf_shap_importance.csv", index=False)
print("\nTop 15 Most Influential Features:")
print(importance_df.head(15).to_string(index=False))


# 4. Plot 1: Global Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Random Forest - Global Feature Importance (Mean |SHAP Value|)")
plt.tight_layout()
plt.savefig("rf_shap_bar.png", dpi=300, bbox_inches="tight")
plt.show()


# 5. Plot 2: Feature Impact Direction & Distribution (Beeswarm Plot)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("Random Forest - SHAP Summary (Log Price Space)")
plt.tight_layout()
plt.savefig("rf_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.show()
