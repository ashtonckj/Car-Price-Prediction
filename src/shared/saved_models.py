"""Save each trained model WITH everything needed to use it on new listings (used by src/external/evaluate_saved_models.py).

One file per model: saved_models/<stage>/<family>_<model>.joblib, e.g. saved_models/b1_tuned/tree_random_forest.joblib.
The file is a dict ("bundle"):
    model          the fitted model (PyTorch networks and TabNet are moved to the CPU first, so the file loads on any computer)
    kind           "sklearn" | "torch" | "tabnet": how to call it (see predict_scaled)
    preprocessing  what the data loader learned from the training rows: imputation tables, one-hot encoders + scaler
                   (ColumnTransformer), target scaler, price range (see core_utils.load_data_full)
    smear_shift    Duan's smearing for THIS model, as a shift in scaled log-price space (0 for a1, which predicts the price)
    + stage, family, model_name, feature_names, test metrics and library versions, for reference.

The files are large (a 500-tree Random Forest is hundreds of MB), so they are stored with Git LFS; rerunning a family
rebuilds them.

Important for custom classes / functions (DCN-V2, KernelRegressor, the KNN inverse_square weights): pickle stores WHERE a
class lives. A script started with the Run button is the module "__main__", and a file saved from there could only be
loaded by that same script. So the family scripts that define such things re-import themselves under their package name
before running main() (see the bottom of run_neural.py and run_distance.py).
"""
import copy
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import full
from src.shared.core_utils import apply_fill, compute_metrics

SAVED_MODELS_ROOT = ROOT_DIR / "saved_models"


def model_path(stage, family, name):
    """saved_models/<stage>/<family>_<model>.joblib, e.g. ("b1_tuned", "tree", "Random Forest") -> .../tree_random_forest.joblib"""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return SAVED_MODELS_ROOT / stage / f"{family}_{slug}.joblib"


def _kind(model):
    if type(model).__name__ == "TabNetRegressor":
        return "tabnet"
    if hasattr(model, "state_dict") and hasattr(model, "parameters"):  # a torch.nn.Module
        return "torch"
    return "sklearn"


def predict_scaled(model, kind, X, batch_size=2048):
    """Model output (scaled log price for a2/b1, price for a1) as a 1-D array, for any kind of model."""
    if kind == "sklearn":
        return np.asarray(model.predict(X)).reshape(-1)
    if kind == "tabnet":
        return np.asarray(model.predict(np.asarray(X, dtype="float32"))).reshape(-1)
    import torch
    device = next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(np.asarray(X, dtype="float32"))
    with torch.no_grad():
        return np.concatenate([model(X[i:i + batch_size].to(device)).cpu().numpy().reshape(-1) for i in range(0, len(X), batch_size)])


def _to_cpu(model, kind):
    """A CPU copy of a PyTorch / TabNet model (the trained model itself is left where it is)."""
    if kind == "torch":
        return copy.deepcopy(model).cpu()
    if kind == "tabnet":
        import torch
        model = copy.deepcopy(model)
        model.network.cpu()
        # TabNet also keeps some tensors as plain attributes (e.g. group_attention_matrix), which .cpu() does not move
        for owner in [model, *model.network.modules()]:
            for attr, value in list(vars(owner).items()):
                if isinstance(value, torch.Tensor):
                    setattr(owner, attr, value.cpu())
        model.device_name, model.device = "cpu", torch.device("cpu")
        return model
    return model


def save_model(stage, family, name, model, split):
    """Save one trained model of a family run (the final model, trained on the 70% train part) with its preprocessing."""
    kind = _kind(model)
    val_pred = predict_scaled(model, kind, split.X_val)
    test_pred = split.smear(val_pred, predict_scaled(model, kind, split.X_test))
    bundle = {
        "stage": stage,
        "family": family,
        "model_name": name,
        "kind": kind,
        "model": _to_cpu(model, kind),
        "preprocessing": split.preprocessing,
        "smear_shift": float(split.smear(val_pred, np.zeros(1))[0]),  # smear(val, y) = y + shift for a2/b1, = y for a1
        "feature_names": split.feature_names,
        "test_metrics": compute_metrics(split.y_test, test_pred, split.inverse_transform_price),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    if kind != "sklearn":
        import torch
        bundle["versions"]["torch"] = torch.__version__
    path = model_path(stage, family, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"  Saved model: {path.relative_to(ROOT_DIR)} ({path.stat().st_size / 1e6:,.1f} MB)")
    return path


def fill_missing(bundle, X):
    """Rows in the columns of cars_full.csv with every missing value filled from the training rows."""
    pre = bundle["preprocessing"]
    X = X[full.FEATURE_COLUMNS].copy()
    X["Condition"] = X["Condition"].fillna(pre["condition_mode"])  # a listing without a condition: the most common one
    for learned in pre["fills"]:  # Fiscal Power, Number of Doors: same group-mode fill as the training rows
        apply_fill(learned, X)
    # Any other missing number (e.g. Equipment Amount, absent from another dataset): the training average, which the
    # fitted scaler stores, so it becomes 0 ("an average car") after scaling
    scaler = pre["preprocessor"].named_transformers_["numeric"]
    return X.fillna(dict(zip(scaler.feature_names_in_, scaler.mean_)))


def predict_from_matrix(bundle, X):
    """Predicted price in MAD (smearing included) from the model's input matrix."""
    y = predict_scaled(bundle["model"], bundle["kind"], X) + bundle["smear_shift"]
    return np.expm1(bundle["preprocessing"]["target_scaler"].inverse_transform(y.reshape(-1, 1)).ravel())

