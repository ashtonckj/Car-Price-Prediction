"""Part B step 3: test the SAVED models (trained on cars_full.csv) on new, real-world listings: data/processed/cars_europe.csv.

Nothing is retrained. Each model is loaded from saved_models/<stage>/<family>_<model>.joblib together with the preprocessing
it learned from the Moroccan training rows, and predicts the European listings (src/preprocessing/external_europe.py:
mobile.de / AutoScout24 cars of the car models the saved models were trained on, priced inside their training price range,
same columns as cars_full.csv, prices in MAD). This answers "does the model work on new data, and what kind of mistakes
does it make?".

A menu lists every saved model that reads cars_full.csv columns (a2_preprocessed, b1_tuned, ...; not a1_baseline, whose
minimal preprocessing has other columns). Type numbers separated by commas, or press Enter for all.

Output (results/b3_external/), like the family files of the other stages: one set per stage and family, and rerunning
some models of a family keeps the others' saved results (the graphs are redrawn from the saved files).
  <stage>_<family>_results.csv           one row per model: R2 / MAPE / MAE / RMSE on the European listings, and the model's
                                         own test metrics on cars_full.csv for comparison
  <stage>_<family>_pred_vs_actual.png    actual vs predicted price, one panel per model of the family
  <stage>_<family>_predictions.npz       the saved predictions that graph is drawn from
  <stage>_<model>_errors.csv             per model: the error per group of cars (age, fuel, fiscal power, brand):
                                         Cars, MAE, MAPE and Bias % = median of (predicted - actual) / actual
                                         (negative = the model predicts too low for that group, positive = too high)
  <stage>_<family>_errors.png            the error tables of the family as bars: bias (top) and MAPE (bottom) per group

Read the results with the differences in mind: another country's market (used cars lose value faster in Germany), no
Condition / Doors (filled with the training values), Fiscal Power filled from the Moroccan data of the same car model.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import external_europe
from src.shared.core_utils import compute_metrics, read_processed
from src.shared.experiment import choose_models
from src.shared.plots import plot_error_groups, plot_family
from src.shared.saved_models import SAVED_MODELS_ROOT, fill_missing, predict_from_matrix

STAGE = "b3_external"
OUT_DIR = ROOT_DIR / "results" / STAGE
SKIPPED_STAGES = {"a1_baseline"}  # minimal preprocessing: other columns
TOP_BRANDS = 10
FAMILY_TITLES = {"linear": "Linear models", "tree": "Tree-based models", "distance": "Distance-based models", "neural": "Neural networks"}
AGE_BANDS = ([-1, 3, 7, 12, 20, 100], ["0-3 y", "4-7 y", "8-12 y", "13-20 y", "21+ y"])
CV_BANDS = ([0, 6, 8, 10, 13, 100], ["<=6 CV", "7-8 CV", "9-10 CV", "11-13 CV", "14+ CV"])
GROUP_ORDERS = {"Age": AGE_BANDS[1], "Fiscal Power": CV_BANDS[1]}  # display order in the error graph


def saved_models():
    """{'b1_tuned / neural_dcn_v2': path} for every saved model that reads cars_full.csv columns."""
    paths = sorted(p for p in SAVED_MODELS_ROOT.glob("*/*.joblib") if p.parent.name not in SKIPPED_STAGES)
    return {f"{p.parent.name} / {p.stem}": p for p in paths}


def as_price(values):
    return np.asarray(values, dtype=float)  # compute_metrics' inverse transform: the prices are already in MAD


def errors_path(saved_model_label):
    """'b1_tuned / neural_mlp' -> results/b3_external/b1_tuned_neural_mlp_errors.csv"""
    return OUT_DIR / f"{saved_model_label.replace(' / ', '_')}_errors.csv"


def error_table(X, actual, predicted):
    """MAE, MAPE and bias per group of cars, to see which kinds of cars the model gets wrong. X: rows with missing values filled."""
    rows = X.assign(error=predicted - actual, pct=(predicted - actual) / actual * 100)
    groups = {
        "Age": pd.cut(rows["Year"].max() - rows["Year"], AGE_BANDS[0], labels=AGE_BANDS[1]),
        "Fuel": rows["Fuel"],
        "Fiscal Power": pd.cut(rows["Fiscal Power (CV)"], CV_BANDS[0], labels=CV_BANDS[1]),
        "Brand": rows["Brand"].where(rows["Brand"].isin(rows["Brand"].value_counts().index[:TOP_BRANDS]), "Other brands"),
    }
    tables = []
    for name, key in groups.items():
        g = rows.groupby(pd.Series(key, index=rows.index).astype(str))
        t = pd.DataFrame({"Cars": g.size(), "MAE": g["error"].apply(lambda e: e.abs().mean()), "MAPE": g["pct"].apply(lambda p: p.abs().mean()), "Bias %": g["pct"].median()})
        tables.append(t.reset_index(names="Value").assign(Group=name))
    out = pd.concat(tables)[["Group", "Value", "Cars", "MAE", "MAPE", "Bias %"]]
    return out.sort_values(["Group", "Cars"], ascending=[True, False]).round(2)


def evaluate(label, bundle, df, actual):
    """Predict, print and save the error table of one model. Returns its results row and predictions."""
    print(f"\n=== {label} ===")
    X = fill_missing(bundle, df)
    predicted = predict_from_matrix(bundle, bundle["preprocessing"]["preprocessor"].transform(X).astype("float32"))
    m = compute_metrics(actual, predicted, as_price)
    test = bundle["test_metrics"]
    print(f"  {len(actual):,} cars: R2 {m['R2']:.3f} | MAPE {m['MAPE']:.1f}% | MAE {m['MAE']:,.0f} MAD")
    print(f"  (its cars_full.csv test set: R2 {test['R2']:.3f} | MAPE {test['MAPE']:.1f}%)")

    error_table(X, actual, predicted).to_csv(errors_path(label), index=False)
    print(f"  Saved {errors_path(label).name}")

    row = {"Saved model": label, "Stage": bundle["stage"], "Family": bundle["family"], "Model": bundle["model_name"],
           "Cars": len(actual), **m, "Test R2 (cars_full)": test["R2"], "Test MAPE (cars_full)": test["MAPE"]}
    return row, predicted


def save_family(stage, family, rows, predictions, actual):
    """Merge the new models into <stage>_<family>_results.csv / _predictions.npz and redraw the family's two graphs.
    predictions: {model name: predicted prices} of the models that were just run."""
    key = OUT_DIR / f"{stage}_{family}"
    results_path, preds_path = Path(f"{key}_results.csv"), Path(f"{key}_predictions.npz")
    table = pd.DataFrame(rows)
    saved = {}
    if results_path.exists():
        old = pd.read_csv(results_path)
        table = pd.concat([old[~old["Model"].isin(table["Model"])], table])
    if preds_path.exists():
        with np.load(preds_path) as npz:  # older predictions are kept if they match the current cars_europe.csv
            saved = {name: npz[name] for name in npz.files if len(npz[name]) == len(actual)}
    saved |= predictions
    table = table.sort_values("R2", ascending=False)
    table.to_csv(results_path, index=False)
    np.savez_compressed(preds_path, **{name: saved[name] for name in table["Model"] if name in saved})
    plot_family_files(stage, family, table, saved, actual)
    print(f"\nSaved {results_path.name}, {key.name}_pred_vs_actual.png and {key.name}_errors.png "
          f"({len(rows)} model(s) updated, {len(table)} in the file)")
    print(table[["Model", "R2", "MAPE", "MAE", "Test R2 (cars_full)"]].round(3).to_string(index=False))


def plot_family_files(stage, family, table, predictions, actual):
    """<stage>_<family>_pred_vs_actual.png and <stage>_<family>_errors.png from the saved results of a family."""
    key = OUT_DIR / f"{stage}_{family}"
    title = f"{FAMILY_TITLES.get(family, family)} ({stage}) on European listings"
    by_model = table.set_index("Model")
    plotted = [name for name in table["Model"] if name in predictions]
    plot_family(title, actual, {name: predictions[name] for name in plotted},
                {name: {"metrics": by_model.loc[name, ["RMSE", "MAE", "R2", "MAPE"]].to_dict()} for name in plotted},
                Path(f"{key}_pred_vs_actual.png"), subset=None)
    tables = {name: pd.read_csv(errors_path(by_model.loc[name, "Saved model"])) for name in table["Model"]
              if errors_path(by_model.loc[name, "Saved model"]).exists()}
    if tables:
        plot_error_groups(title, tables, Path(f"{key}_errors.png"), orders=GROUP_ORDERS)


def main():
    models = saved_models()
    if not models:
        sys.exit("No saved models found: run a family in src/models/ or src/tuned_models/ first (it saves to saved_models/).")
    chosen = choose_models(list(models), "test on the European listings")

    df = read_processed(external_europe.OUT_PATH, external_europe.main)
    actual = np.expm1(df["Log Price"].to_numpy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"European listings: {len(df):,} cars, median price {np.median(actual):,.0f} MAD")

    families = {}  # (stage, family) -> (rows, {model name: predictions})
    for label in chosen:
        bundle = joblib.load(models[label])
        row, predicted = evaluate(label, bundle, df, actual)
        rows, predictions = families.setdefault((bundle["stage"], bundle["family"]), ([], {}))
        rows.append(row)
        predictions[bundle["model_name"]] = predicted

    for (stage, family), (rows, predictions) in families.items():
        save_family(stage, family, rows, predictions, actual)


if __name__ == "__main__":
    main()
