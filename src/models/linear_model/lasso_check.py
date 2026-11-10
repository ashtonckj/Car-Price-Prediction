import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.core_utils import PROCESSED_CSV, RANDOM_STATE, load_data

ALPHA = 1.0  # Lasso's default
X_train, y_train, *_ = load_data()


df = pd.read_csv(PROCESSED_CSV)
X = df.drop(columns="Log Price")
X_tr, _ = train_test_split(X, test_size=0.2, random_state=RANDOM_STATE)
X_tr, _ = train_test_split(X_tr, test_size=0.125, random_state=RANDOM_STATE)
numeric = [c for c in X.columns if c not in ["Brand", "Model", "Fuel"]]
names = ColumnTransformer([
    ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"), ["Brand", "Model"]),
    ("fuel", OneHotEncoder(handle_unknown="ignore"), ["Fuel"]),
    ("numeric", "passthrough", numeric),
]).fit(X_tr).get_feature_names_out()
if len(names) != X_train.shape[1]:
    names = [f"feature_{i}" for i in range(X_train.shape[1])]  

y = y_train.ravel().astype(float)
y_centred = y - y.mean()
strength = np.abs(X_train.astype(float).T @ y_centred) / len(y)

table = (pd.DataFrame({"Feature": names, "Strength": strength})
         .sort_values("Strength", ascending=False).reset_index(drop=True))
table["Is > alpha=1.0?"] = np.where(table["Strength"] > ALPHA, "Yes", "No")

print(f"\nLasso alpha: {ALPHA}")
print("\nTop 15 strongest features:")
print(table.head(15).to_string(float_format="{:.4f}".format))
print(f"\nFeatures stronger than alpha: {(table['Strength'] > ALPHA).sum()} of {len(table)}")
print(f"Strongest feature: {table.loc[0, 'Feature']} = {table.loc[0, 'Strength']:.4f}")