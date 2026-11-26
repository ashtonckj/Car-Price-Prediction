"""
SHAP analysis for specific models (Decision Tree, Random Forest, Gradient Boosting, KNN, PyTorch MLP, DCN-V2).

Run this script and it will:
  1. Ask which Part A stage (a1_baseline / a2_preprocessed)
  2. Ask which model
  3. Train the model on the final split
  4. Compute SHAP values on the test set using the appropriate explainer:
       - TreeExplainer for tree models (Decision Tree, Random Forest, Gradient Boosting)
       - Explainer with background dataset for KNN, PyTorch MLP, and DCN-V2 (from neural_models.py)
  5. Save to  results/<stage>/shap/<model_key>/
        <model_key>_shap_bar.png
        <model_key>_shap_beeswarm.png
        <model_key>_shap_importance.csv
        <model_key>_shap_values.joblib
        <model_key>_model.joblib

The data comes from the same loaders as the model scripts (src/shared/stages.py -> core_utils.load_data_*), so SHAP
always sees exactly the rows, price scope and features the models were scored on. Feature names come from the loader
(e.g. "Brand_Toyota", "Fuel_Diesel").

No smearing here: SHAP explains the model output (scaled log price for a2), and smearing only adds the same constant to
every prediction, which would move the base value but not change any SHAP value or the feature ranking.
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.neural_models import DCNV2, build_mlp, train_torch
from src.shared.core_utils import RANDOM_STATE
from src.shared.stages import STAGES, choose_stage, load_data

MODELS = {
    "decision_tree": {
        "title": "Decision Tree",
        "make": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    },
    "random_forest": {
        "title": "Random Forest",
        "make": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    },
    "gradient_boosting": {
        "title": "Gradient Boosting",
        "make": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    },
    "knn": {
        "title": "K-Nearest Neighbors",
        "make": lambda: KNeighborsRegressor(n_jobs=-1),
    },
    "mlp": {
        "title": "Multi-Layer Perceptron (PyTorch)",
        "make": build_mlp,  # same networks as src/models/neural_models.py
    },
    "dcn_v2": {
        "title": "Deep & Cross Network V2 (PyTorch)",
        "make": DCNV2,
    },
}

TREE_MODELS = {"decision_tree", "random_forest", "gradient_boosting"}
PYTORCH_MODELS = {"mlp", "dcn_v2"}


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

    print(f"\n[1/4] Loading data  ({stage['title']}) ...")
    split = load_data(stage_key)  # same rows, price scope and features as the model scripts
    X_train, X_val, X_test = split.X_train, split.X_val, split.X_test
    y_train, y_val = split.y_train.ravel(), split.y_val.ravel()
    feature_names = split.feature_names

    print(f"      Features: {len(feature_names)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print(f"\n[2/4] Training  {model_cfg['title']} ...")
    if model_key in PYTORCH_MODELS:
        # PyTorch models (MLP, DCN-V2) from src/models/neural_models.py
        model = train_torch(model_cfg["make"](X_train.shape[1]), X_train, y_train.reshape(-1, 1), X_val, y_val.reshape(-1, 1),
                            epochs=100, patience=10)

        def predict_fn(X_batch):
            device = next(model.parameters()).device
            model.eval()
            arr = np.array(X_batch, dtype=np.float32, copy=True)  # contiguous, writable float32 for PyTorch
            with torch.no_grad():  # in batches: the permutation explainer evaluates many rows at once
                return np.concatenate([model(torch.as_tensor(arr[i:i + 1024], device=device)).cpu().numpy().ravel()
                                       for i in range(0, len(arr), 1024)])

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

    # Subsample the test set: 500 rows for tree models (~1-2 min); 100 for the others, whose permutation explainer
    # needs many forward passes per row with 300+ features (500 would be ~5x slower for nearly the same ranking)
    eval_sample_size = 500 if model_key in TREE_MODELS else 100
    if len(X_test_df) > eval_sample_size:
        print(f"Subsampling test set to {eval_sample_size} samples for fast SHAP computation...")
        X_test_sample = X_test_df.sample(n=eval_sample_size, random_state=RANDOM_STATE)
    else:
        X_test_sample = X_test_df

    print("\n[3/4] Computing SHAP values ...")
    if model_key in TREE_MODELS:
        # A1 has a high-dimensional one-hot matrix and unrestricted trees: an explicit training background gives
        # stable interventional SHAP values
        background = shap.sample(X_train_df, 100, random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="raw")
        shap_values = explainer(X_test_sample, check_additivity=True)
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
    ax.set_title(f"SHAP Feature Importance (bar)  —  {model_cfg['title']}  [{stage_key}]")
    fig.tight_layout()
    bar_path = out_dir / f"{model_key}_shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {bar_path.name}")

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=20, show=False, ax=ax, plot_size=None)
    ax.set_title(f"SHAP Beeswarm  —  {model_cfg['title']}  [{stage_key}]")
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

    print("\nDone! Top-5 features by mean |SHAP|:")
    print(importance_df.head(5).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    stage_key = choose_stage("A")
    model_key = choose_model()
    run_shap(stage_key, model_key)


if __name__ == "__main__":
    main()
