"""
SHAP analysis for specific models (Decision Tree, Random Forest, Gradient Boosting, KNN, PyTorch MLP).

Run this script and it will:
  1. Ask which stage (a1_baseline / a2_preprocessed)
  2. Ask which model
  3. Train the model on the final split
  4. Compute SHAP values on the test set using the appropriate explainer:
       - TreeExplainer for tree models (Decision Tree, Random Forest, Gradient Boosting)
       - Explainer with background dataset for KNN and PyTorch MLP (from neural_models.py)
  5. Save to  results/<stage>/shap/<model_key>/
        <model_key>_shap_bar.png
        <model_key>_shap_beeswarm.png
        <model_key>_shap_importance.csv
        <model_key>_shap_values.joblib
        <model_key>_model.joblib

Feature names come directly from the CSV columns (OHE categories use the
exact CSV text, e.g. "Brand_Toyota", "Fuel_Diesel").
"""
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.neural_models import train_torch
from src.shared.core_utils import RANDOM_STATE, fill_from_train, split_rows
from src.preprocessing import full, minimal

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def make_mlp(n_features: int) -> nn.Module:
    """PyTorch MLP architecture matching src/models/neural_models.py."""
    return nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


MODELS = {
    "decision_tree": {
        "title": "Decision Tree",
        "make": lambda n_features=None: DecisionTreeRegressor(random_state=RANDOM_STATE),
    },
    "random_forest": {
        "title": "Random Forest",
        "make": lambda n_features=None: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    },
    "gradient_boosting": {
        "title": "Gradient Boosting",
        "make": lambda n_features=None: GradientBoostingRegressor(random_state=RANDOM_STATE),
    },
    "knn": {
        "title": "K-Nearest Neighbors",
        "make": lambda n_features=None: KNeighborsRegressor(n_jobs=-1),
    },
    "mlp": {
        "title": "Multi-Layer Perceptron (PyTorch)",
        "make": make_mlp,
    },
}

TREE_MODELS = {"decision_tree", "random_forest", "gradient_boosting"}
BRAND_MODEL_COLS = ["Brand", "Model"]


# ---------------------------------------------------------------------------
# Data loading with feature-name reconstruction
# ---------------------------------------------------------------------------
def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract clean feature names from a fitted ColumnTransformer.

    Uses preprocessor.get_feature_names_out() and strips transformer prefixes
    (e.g., 'text__', 'remainder__', 'numeric__') so names match the exact CSV features.
    """
    return [name.split("__", 1)[-1] if "__" in name else name for name in preprocessor.get_feature_names_out()]


def load_minimal_with_names():
    """Load a1_baseline data and return (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)."""
    df = pd.read_csv(minimal.OUT_PATH)
    X, y = df.drop(columns="Price"), df[["Price"]]
    X_train, y_train, X_val, y_val, X_test, y_test = split_rows(X, y)

    text_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        [("text", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), text_cols)],
        remainder="passthrough",
    )
    X_train_t = preprocessor.fit_transform(X_train).astype("float32")
    X_val_t = preprocessor.transform(X_val).astype("float32")
    X_test_t = preprocessor.transform(X_test).astype("float32")
    y_train_arr = y_train.to_numpy().astype("float32").ravel()
    y_val_arr = y_val.to_numpy().astype("float32").ravel()
    y_test_arr = y_test.to_numpy().astype("float32").ravel()

    feature_names = _get_feature_names(preprocessor)
    return X_train_t, y_train_arr, X_val_t, y_val_arr, X_test_t, y_test_arr, feature_names


def load_full_with_names():
    """Load a2_preprocessed data and return (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)."""
    df = pd.read_csv(full.OUT_PATH)
    X, y = df.drop(columns="Log Price"), df[["Log Price"]]
    X_train, y_train, X_val, y_val, X_test, y_test = split_rows(X, y)
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    fill_from_train("Fiscal Power (CV)", [BRAND_MODEL_COLS + ["Fuel"], BRAND_MODEL_COLS], None, X_train, X_val, X_test)
    fill_from_train("Number of Doors", [BRAND_MODEL_COLS], 5, X_train, X_val, X_test)

    numeric = [c for c in X_train.columns if c not in BRAND_MODEL_COLS + ["Fuel"]]

    preprocessor = ColumnTransformer([
        ("brand_model", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), BRAND_MODEL_COLS),
        ("fuel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel"]),
        ("numeric", StandardScaler(), numeric),
    ])
    X_train_t = preprocessor.fit_transform(X_train).astype("float32")
    X_val_t = preprocessor.transform(X_val).astype("float32")
    X_test_t = preprocessor.transform(X_test).astype("float32")

    target_scaler = StandardScaler().fit(y_train)
    y_train_arr = target_scaler.transform(y_train).astype("float32").ravel()
    y_val_arr = target_scaler.transform(y_val).astype("float32").ravel()
    y_test_arr = target_scaler.transform(y_test).astype("float32").ravel()

    feature_names = _get_feature_names(preprocessor)
    return X_train_t, y_train_arr, X_val_t, y_val_arr, X_test_t, y_test_arr, feature_names


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
STAGES = {
    "a1_baseline": {
        "title": "A1 baseline (minimal preprocessing)",
        "load": load_minimal_with_names,
        "dataset": "cars_minimal.csv",
    },
    "a2_preprocessed": {
        "title": "A2 full preprocessing",
        "load": load_full_with_names,
        "dataset": "cars_full.csv",
    },
}


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------
def choose_stage() -> str:
    keys = list(STAGES)
    print("\nWhich stage / dataset do you want to use?")
    for i, key in enumerate(keys, 1):
        s = STAGES[key]
        print(f"  {i} = {key:<18}  {s['title']}  [{s['dataset']}]")
    choice = input("Type a number and press Enter: ").strip()
    if not (choice.isdigit() and 1 <= int(choice) <= len(keys)):
        sys.exit(f"'{choice}' is not a valid option.")
    return keys[int(choice) - 1]


def choose_model() -> str:
    keys = list(MODELS)
    print("\nWhich model do you want to explain with SHAP?")
    for i, key in enumerate(keys, 1):
        print(f"  {i} = {MODELS[key]['title']}")
    choice = input("Type a number and press Enter: ").strip()
    if not (choice.isdigit() and 1 <= int(choice) <= len(keys)):
        sys.exit(f"'{choice}' is not a valid option.")
    return keys[int(choice) - 1]


# ---------------------------------------------------------------------------
# SHAP computation + saving
# ---------------------------------------------------------------------------
def run_shap(stage_key: str, model_key: str) -> None:
    stage = STAGES[stage_key]
    model_cfg = MODELS[model_key]

    print(f"\n[1/4] Loading data  ({stage['dataset']}) ...")
    if stage_key == "a1_baseline":
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_minimal_with_names()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_full_with_names()

    print(f"      Features: {len(feature_names)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print(f"\n[2/4] Training  {model_cfg['title']} ...")
    if model_key == "mlp":
        # PyTorch MLP from src/models/neural_models.py
        model = make_mlp(n_features=X_train.shape[1])
        model = train_torch(
            model,
            X_train,
            y_train.reshape(-1, 1),
            X_val,
            y_val.reshape(-1, 1),
            epochs=100,
            patience=10,
        )

        def predict_fn(X_batch):
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                # Copy to ensure a contiguous, writable array for PyTorch
                arr = np.array(X_batch, dtype=np.float32, copy=True)
                # Batch evaluation to avoid high memory usage during permutation evaluation
                batch_size = 1024
                preds = []
                for i in range(0, len(arr), batch_size):
                    tensor = torch.as_tensor(arr[i : i + batch_size], dtype=torch.float32, device=device)
                    preds.append(model(tensor).detach().cpu().numpy().ravel())
                return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    else:
        # Scikit-learn models (Tree-based or KNN)
        model = model_cfg["make"]()
        model.fit(X_train, y_train)

        def predict_fn(X_batch):
            # Pass as raw numpy array to prevent scikit-learn feature name mismatch warnings
            return model.predict(np.asarray(X_batch))

    # Wrap arrays in DataFrames so SHAP uses exact feature names
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    # Subsample test set for fast SHAP calculation
    # Tree models: 500 samples (~1-2 min)
    # Non-tree models (MLP, KNN with 300+ features): 50-100 samples
    '''(100 samples) is recommended because with 300+ features, using 500 background samples 
    will cause the explanation to evaluate 5 times more forward passes, making it noticeably 
    slower with virtually no difference in the resulting plots or rankings.'''
    if model_key in TREE_MODELS:
        eval_sample_size = 500
    else:
        eval_sample_size = 100

    if eval_sample_size and len(X_test_df) > eval_sample_size:
        print(f"Subsampling test set to {eval_sample_size} samples for fast SHAP computation...")
        X_test_sample = X_test_df.sample(n=eval_sample_size, random_state=RANDOM_STATE)
    else:
        X_test_sample = X_test_df

    print(f"\n[3/4] Computing SHAP values ...")
    if model_key in TREE_MODELS:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_test_sample)
    else:
        print("Creating background summary dataset from training set (100 samples)...")
        background = shap.sample(X_train_df, 100, random_state=RANDOM_STATE)
        explainer = shap.Explainer(predict_fn, background)
        # PermutationExplainer requires at least 2 * num_features + 1 evaluations per sample
        min_evals = 2 * len(feature_names) + 1
        print(f"Running PermutationExplainer with max_evals={min_evals} across {len(X_test_sample)} samples...")
        shap_values = explainer(X_test_sample, max_evals=min_evals)

    out_dir = ROOT_DIR / "results" / stage_key / "shap" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Saving outputs to  {out_dir} ...")

    # --- Bar plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=20, show=False, ax=ax)
    ax.set_title(f"SHAP Feature Importance (bar)  —  {model_cfg['title']}  [{stage['dataset']}]")
    fig.tight_layout()
    bar_path = out_dir / f"{model_key}_shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {bar_path.name}")

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=20, show=False, ax=ax, plot_size=None)
    ax.set_title(f"SHAP Beeswarm  —  {model_cfg['title']}  [{stage['dataset']}]")
    fig.tight_layout()
    beeswarm_path = out_dir / f"{model_key}_shap_beeswarm.png"
    fig.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {beeswarm_path.name}")

    # --- Importance CSV (mean |SHAP| per feature, sorted descending) ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.index += 1  # rank starts at 1
    importance_df.index.name = "Rank"
    csv_path = out_dir / f"{model_key}_shap_importance.csv"
    importance_df.to_csv(csv_path)
    print(f"  Saved: {csv_path.name}")

    # --- joblib: SHAP values ---
    shap_joblib_path = out_dir / f"{model_key}_shap_values.joblib"
    joblib.dump(shap_values, shap_joblib_path)
    print(f"  Saved: {shap_joblib_path.name}")

    # --- joblib: Trained Model ---
    model_joblib_path = out_dir / f"{model_key}_model.joblib"
    joblib.dump(model, model_joblib_path)
    print(f"  Saved: {model_joblib_path.name}")

    print(f"\nDone! Top-5 features by mean |SHAP|:")
    print(importance_df.head(5).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    stage_key = choose_stage()
    model_key = choose_model()
    run_shap(stage_key, model_key)


if __name__ == "__main__":
    main()
