"""
train_xgb.py
Loads data/processed/cars.csv, cleans and imputes Mileage, and trains XGBRegressor.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "cars.csv"


def parse_mileage(val):
    """Parses mileage ranges ('10000 - 20000') or strings into floats."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace(" ", "").replace("\xa0", "").strip()
    if "-" in val_str:
        parts = val_str.split("-")
        nums = [re.sub(r"[^\d]", "", p) for p in parts if re.sub(r"[^\d]", "", p)]
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2.0
    num_only = re.sub(r"[^\d]", "", val_str)
    return float(num_only) if num_only else np.nan


def main():
    # 1. Load data from processed CSV
    df = pd.read_csv(PROCESSED_CSV)

    # 1.1 Check for duplicates and remove them
    print(f"Duplicate rows: {df.duplicated().sum()}")
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicate rows removed. New shape: {df.shape}")

    # 2. Parse Mileage & apply 3-tier hierarchical median imputation
    if "Mileage" in df.columns:
        df["Mileage"] = df["Mileage"].apply(parse_mileage)

    # 3. Separate Features and Log-Transformed Target (Handle Skewed Price Distribution)
    y = np.log1p(df["Price"])
    X = df.drop(columns=["Price", "First Owner"])

    # 4. Categorical and Numerical Feature Split
    cat_cols = [col for col in ["Brand", "Model", "Fiscal Power"] if col in X.columns]
    num_cols = [col for col in X.columns if col not in cat_cols]

    # 5. Preprocessing & XGBoost Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                cat_cols,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    random_state=42
                ),
            ),
        ]
    )

    # 6. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Model Training & Evaluation
    print("Training XGBoost Regressor...")
    model_pipeline.fit(X_train, y_train)

    y_pred_log = model_pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    # R2 in Log Space (True Model Performance)
    r2_log = r2_score(y_test, y_pred_log)

    print("\n--- Model Performance ---")
    print(f"MAE:  {mae:,.2f} MAD")
    print(f"R²:   {r2:.4f}")
    print(f"Log-scale R² (Actual): {r2_log:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 8. Feature Importances
    ohe_cols = (
        model_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
    all_features = list(num_cols) + list(ohe_cols)
    importances = model_pipeline.named_steps["regressor"].feature_importances_

    top_features = (
        pd.Series(importances, index=all_features)
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 Feature Importances:")
    print(top_features.to_string())


if __name__ == "__main__":
    main()