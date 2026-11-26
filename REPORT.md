# What the report must say

The points the written report **has to** explain, with the reasoning and the numbers behind them. All numbers come from
the saved files in `results/`; re-run a script before quoting it if its code or data changed.

---

## 1. Which rows are used (data cleaning, before the split)

Applied to **every row before the train/test split** (`src/preprocessing/full.py`; `minimal.py` keeps the same rows, so
both Part A stages start from the same cars):

| Rule | Rows removed | Reason to give |
|---|---|---|
| No price | 25,483 | A target value is never invented |
| Price below 5,000 MAD | 557 | Not real asking prices. Even between 3,000 and 5,000 MAD, 98% of listings are under a fifth of the usual price of their model while being listed in good condition |
| Price above 2,000,000 MAD | 211 | Mostly typos and placeholder values; the few real ones are too rare to learn from |
| Placeholder prices (one digit repeated, e.g. 11111, 999999) | 12 | Example: a 2011 Dacia Logan at 999,999 MAD while that model's median is 85,000. 9,999 and 99,999 are kept: they are normal "just under 10k / 100k" prices |
| **Price far from similar cars** (`drop_price_outliers`) | 1,027 | See below |
| Missing Condition | 2,125 | Cannot be encoded |
| Duplicate listings | 11,126 | Would be counted twice |
| Corrupted row (a mileage in the Fuel column) | 1 | Unusable |

101,896 raw listings → **61,354 rows**.

**The "price far from similar cars" rule** must be explained properly, because it is the one rule that is not fixed in advance:

- **What it does.** Older cars are cheaper, so the general year trend is first taken out of the log price. Each car is then
  compared with the median of its Brand + Model (or its Brand, if the model has fewer than 5 listings). A car is removed if its
  price is more than 1.5 away in log price, i.e. about **4.5 times** too high or too low for that kind of car.
- **Why.** It catches wrong prices that a fixed minimum or maximum cannot see, e.g. a 2021 Dacia Sandero at 1,550,000 MAD or
  a 2011 Volkswagen Touareg at 3,000 MAD.
- **Say openly that this is a mild form of leakage.** The medians and the year trend are calculated from all rows, test rows
  included, and the rule can remove a test car *because of its price*, which could not be done for a new car whose price is
  unknown. Our lecturer agreed this is acceptable for this assignment. A leakage-free version would learn the medians and
  the trend from the training rows only and filter only the training rows.

---

## 2. The price range: the model is for cars between about 21,000 and 445,000 MAD

After the split, only cars whose price lies between the **2nd and 98th percentile of the training rows** are kept, in the
training, validation **and** test sets alike (`PRICE_QUANTILES` in `src/shared/core_utils.py`). On the final split that is
21,000–445,000 MAD. This is done in `a2_preprocessed` and `b1_tuned`, not in the baseline (section 3).

- **It is a scope, not cleaning.** It removes real cars at both ends: mostly luxury brands (Mercedes-Benz, Audi, Land
  Rover, Porsche) at the top and very old cars at the bottom; 440 of the 12,271 test cars (3.6%). Say plainly that the model
  is built for this price range and is not meant for luxury or scrap-value cars.
- **Why it is needed.** R2 and RMSE come from squared errors in MAD, so a handful of very expensive cars dominate them. In an
  earlier version without a range, 50 test cars out of 12,464 (0.4%) made up 70% of all squared error; for example a car
  listed at 2,900,000 MAD was predicted at about 300,000.
- **Why q2–q98 and not q1–q99.** Both were tested; q2–q98 gave about +0.06 R2 more, at the cost of a narrower range of cars.
- **Why it is not leakage.** The two limits are learned from the training rows only (in cross-validation, from each fold's
  own training rows) and then applied to the other sets, just like a scaler.
- **Why test rows are filtered too.** The model is trained and scored on the same range of cars. Trimming only the
  training rows was tried first: the model could then never predict above about 650k while the test set still contained
  cars up to 2.9M, and it scored *worse* (Polynomial test R2 0.591 vs 0.614).

---

## 3. Baseline (a1) vs full preprocessing (a2)

The baseline does the minimum: same rows as section 1, but **every** price is kept after the split (no q2–q98 range),
missing numbers are filled with 0 and missing text with "missing", every raw column is one-hot encoded and the raw price is
predicted. The full stage adds the price range, cleans the car names, converts mileage ranges to numbers, imputes missing
values from similar cars, scales the numbers and predicts the log price.

| Model | a1 CV R2 | a2 CV R2 | a1 test MAE | a2 test MAE | a1 MAPE | a2 MAPE |
|---|---|---|---|---|---|---|
| Ridge | 0.786 | 0.872 | 28,134 | 16,906 | 28.5% | 15.0% |
| Polynomial Regression | 0.785 | 0.886 | 28,165 | 15,851 | 28.5% | 14.0% |
| Random Forest | 0.834 | 0.913 | 19,370 | 13,714 | 16.6% | 12.4% |
| KNN | 0.756 | 0.863 | 27,168 | 17,648 | 26.5% | 15.9% |
| Radius Neighbours | 0.005 | 0.710 | 72,544 | 24,656 | 97.5% | 25.3% |
| MLP | 0.880 | 0.920 | 17,421 | 13,548 | 15.6% | 12.2% |
| DCN-V2 | 0.899 | 0.917 | 16,533 | 13,443 | 14.5% | 12.0% |

- **The dataset is feasible:** with minimal preprocessing most models already explain 71–90% of the price variance
  (CV R2); the exceptions are Radius Neighbours (below) and TabNet (0.64 ± 0.11, unstable on the raw features).
- **The two stages are not scored on the same cars** (a1: 12,271 test cars, a2: 11,831), because removing the cheapest and
  most expensive cars is itself part of the full preprocessing. So the a1 → a2 gain has two causes that must both be named:
  (1) the price range removes the cars that are hardest to predict, and (2) the column preprocessing (names, imputation,
  scaling, log target). An earlier check with both stages on exactly the same cars showed that the column preprocessing
  alone lowers the typical error by about a quarter (linear models: MAE about 26k → 19–21k MAD), while R2 rose only 0.02–0.05.
- **Radius Neighbours in a1 (R2 0.005) is not a bug, and it shows why scaling matters.** It averages the cars within a
  fixed distance of 1.0. Without scaling, Year and Mileage are in years and kilometres, so almost no car has another car that
  close, and it falls back to the average price for nearly every car, which gives R2 ≈ 0. With scaled features (a2) the
  same model reaches 0.71.

---

## 4. Target: log price, converted back with a smearing correction

- **Why log price:** prices are strongly right-skewed and errors grow with the price, so the full stage predicts
  log(1 + price). With the raw price as target, test R2 fell for every linear model (Ridge 0.541 → 0.495, Polynomial
  0.591 → 0.556; measured on an earlier version of the data). A square-root target landed between the two.
- **All metrics are on the price in MAD**, never on the log price, so every stage and model is directly comparable.
- **Smearing (Duan, 1983).** Converting a log prediction back with `exp` gives roughly the *median* price for those
  features, which on average underestimates the price. Every predicted price is multiplied by `mean(exp(residual))`, where
  the residuals are the log-price errors on the **validation** rows (never the test rows). It raised R2 on every CV fold
  (Ridge +0.011 to +0.016, Polynomial +0.001 to +0.010; earlier version of the data) and makes little difference for
  Random Forest. It does nothing for the baseline, which predicts the price directly.

---

## 5. How leakage is handled (summary paragraph)

- The data is split **70% train / 10% validation / 20% test**. The test set is scored once at the end.
- Imputation values, one-hot categories, scalers and the q2–q98 limits are learned from the training rows only; the
  smearing factor from the validation rows.
- Model selection and tuning use the validation rows or 5-fold cross-validation on the 70% train part, never the test set.
- Test rows are never removed because the model gets them wrong. A stricter version of the "price far from similar cars"
  rule (3× or 5× from the model median) was tried and dropped: it only raised the score by deleting hard test cars.
- Before the split, fixed rules (section 1) are applied identically to all rows. **The one exception** is the "price far
  from similar cars" rule, which uses statistics of all rows: state it as a known, accepted limitation.

---

## 6. Overfitting check (complexity curves)

`src/analysis/a2_preprocessed_complexity_curve.py` → `results/a2_preprocessed/complexity_curve.png` (default models) and
`src/analysis/b1_tuned_complexity_curve.py` → `results/b1_tuned/complexity_curve.png` (tuned models);
`complexity_comparison.py` puts both side by side in `results/comparison/complexity_comparison.png`. For one model per
family, one setting that controls complexity is swept while the others stay at their default (a2) or tuned (b1) values,
and the error on the training rows is compared with the error on the validation rows. The test rows are not used.

- **Polynomial Regression (degree):** degree 2–4 are close (a2 validation RMSE 26.1–27.5k MAD), then the model explodes
  from degree 5 (validation errors in the millions and billions: too many terms for the data). With the tuned settings,
  validation error already rises from degree 2 (27.1k) to 4 (33.7k) while training error stays at 26.4k: clear
  overfitting. The tuned degree 2 is the right choice.
- **Random Forest (max_depth):** validation RMSE falls from 29.4k (depth 10) to 22.4k at depth 40 and stays flat after,
  because the trees are fully grown by then. Training error (9.8k) is far below validation, but validation never rises:
  averaging many trees keeps the deep trees from overfitting. The tuned depth 50 is on the plateau.
- **KNN (k):** k = 1 is the textbook overfit (a2: training 4.9k, validation 34.2k: every car is its own nearest
  neighbour); very large k underfits (both about 35k). a2 is best near k = 5 (28.4k); tuned KNN (inverse-square distance
  weights, Manhattan distance) is best at k = 9–15 (26.7k), and the tuned k = 9 is there. With distance weights the
  training error is near zero for every k, so only the validation curve is informative.
- **DCN-V2 (training epochs):** trained with the same early stopping as the real runs (patience 10). Validation error
  stops improving at epoch 15 in a2 (22.6k vs 19.6k training) and epoch 13 in b1 (21.9k vs 19.1k); training then stops
  10 epochs later and the weights of the best epoch are kept, which is what prevents the network from overfitting.
- **Which neural network:** the **MLP** is carried forward (ensembles) rather than DCN-V2. On cross-validation the MLP is
  ahead at both stages (a2: 0.920 vs 0.917; tuned: 0.920 vs 0.919); the differences are smaller than the CV standard
  deviation (0.005–0.009), so they are a tie, and the simpler model is preferred.
- **Be honest about tuning gains:** tuning did not change the MLP's CV R2 (0.920 → 0.920); its test gain (0.921 → 0.930) is
  within the noise of a single split. Tuning helped most where the defaults were poor: Gradient Boosting (CV 0.856 → 0.923),
  Radius Neighbours (0.710 → 0.865), Decision Tree (0.853 → 0.887) and KNN (0.863 → 0.882).

---

## 7. How to read the results

- **Report CV mean ± std next to the test score.** The test score depends on which cars happen to be in the 20%:
  changing the data by even a few rows reshuffles the split and can move test R2 by about ±0.03. Preprocessing decisions
  were made on CV R2 for this reason, and the CV standard deviation shows whether a result is stable.
- **Report MAE and MAPE next to R2:** R2 and RMSE are dominated by the most expensive cars.

---

## 8. Limitations to state

- **Price range:** only cars between about 21k and 445k MAD; luxury and very cheap cars are out of scope.
- **The "price far from similar cars" rule** uses all rows, test rows included (section 1).
- **The neural networks** also use the validation rows for early stopping, so for them the smearing factor is learned on
  rows that already influenced training. The effect is small, but say so in one sentence.
- **The data is from 2024 listings**; asking prices drift over time, which matters for the new-data test.
- **New-data test (section 10):** the external cars have no Condition or Doors (filled with training values), their fiscal
  power and equipment are approximated, and only car models the model was trained on, priced inside its training price
  range, are tested: the model makes no claim for other cars.
- **Asking prices, not sale prices:** listings contain negotiation margin and some wrong values that no rule can catch.

---

## 9. Ensembles (Part B step 2, `b2_ensemble`)

`src/ensemble/` combines tuned b1 models; results in `results/b2_ensemble/`. Three combinations, each combined in two ways:

- **Stacking:** each base model is trained 5 times on 4/5 of the training rows and predicts the remaining fifth, so every
  training car gets a prediction from a model that did not see it ("out-of-fold"). A Ridge regression learns how to weigh
  the base models from those predictions.
- **Blending:** the same out-of-fold predictions, but the weights are found directly (between 0 and 1, adding up to 1) to
  give the lowest RMSE. The weights say how much each model counts.

| Ensemble | Test R2 | MAE (MAD) | MAPE |
|---|---|---|---|
| Polynomial + Gradient Boosting + Kernel Regression + MLP | **0.934** | 12,765 | 11.5% |
| Random Forest + Gradient Boosting | 0.932 | 12,950 | 11.6% |
| MLP + DCN-V2 | 0.931 | 12,940 | 11.7% |
| *Best single b1 model: MLP* | *0.930* | *13,176* | *12.0%* |

Stacking and blending give the same scores (to 3 decimals), so the table shows one row per combination.

- **Compare with the best single b1 model**, not in the a1 / a2 / b1 stage comparison (an ensemble is not the same models
  under a new setting).
- **The one-model-per-family combination gains most**, as expected: its models make *different* mistakes, while RF + GB
  and MLP + DCN-V2 are similar to each other.
- **Be honest about the size of the gain:** +0.004 R2 and about 3% lower MAE than the best single model, on one test
  split and without cross-validation (it would mean 5 × 5 fits per base model). That is within the ±0.03 test-split noise
  of section 7, at a much higher cost: about 66 minutes of training for the four-model ensemble against 35 seconds for the MLP.
- The networks stop early on the 10% validation rows, never on the out-of-fold rows, so the out-of-fold predictions the
  weights are learned from stay honest.

---

## 10. Do the saved models work on new data? (Part B step 3, `b3_external`)

The spec asks to "measure how well the model performs on new data" and to "look at the types of mistakes it makes". We test
the saved models on real-world used-car listings from another market: KTHESAT, "European Used Cars 100k Sample", Hugging
Face, CC BY-NC 4.0 (100,000 mobile.de / AutoScout24 listings, mostly Germany). The repository is public, so scraped
website listings are not used.

**Nothing is retrained.** `src/external/evaluate_saved_models.py` loads each model saved from `cars_full.csv` (with the
imputation, encoders, scaler, target scaler and smearing learned from the Moroccan training rows) and predicts the listings.
`src/preprocessing/external_europe.py` gives them the same columns as `cars_full.csv`:
- **Price** converted to MAD at one fixed rate (1 EUR = 10.77 MAD, the 2026 average) and rounded to 10 MAD; the same price
  rules as `full.py`. One rate for every car does not change R2 or MAPE.
- **Car names** written as in the Moroccan data (BMW 320 → serie 3, Mercedes C 220 → classe c, MINI → Mini ...).
- **Equipment Amount:** how many of the 16 Moroccan equipment items the listing's (German) equipment list contains.
- **Mileage** in km put into the training data's ranges, then the range midpoint, as in `full.py`.
- **Fiscal Power (CV):** the listings give horsepower, while the Moroccan CV is based on engine size and fuel, so it is
  filled like any missing CV: the Moroccan CV of the same Brand + Model + Fuel.
- **Condition and Number of Doors** are not in the dataset: most common training condition, training Brand/Model doors.

**Which cars.** Only the cars the saved models are made for: car models they were trained on (Brand + Model pairs with at
least 10 cars in the Moroccan training rows, the encoder's minimum, so each has its own column) priced inside the training
price range (21k–445k MAD, the same rule as the Moroccan test set, so the scores are comparable). Both rules come from the
training rows only, never from the European prices or the errors. That leaves 45,995 listings.

Results of the b1 models (`results/b3_external/b1_tuned_<family>_results.csv` and `_pred_vs_actual.png`):

| Model | R2 | MAPE | MAE (MAD) | Test R2 on cars_full.csv | Test MAPE on cars_full.csv |
|---|---|---|---|---|---|
| MLP | **0.713** | 32.5% | 42,306 | 0.930 | 12.0% |
| Gradient Boosting | 0.666 | 33.1% | 45,318 | 0.929 | 11.9% |
| Ridge | 0.658 | 34.4% | 45,744 | 0.878 | 15.0% |
| Huber | 0.656 | 34.7% | 45,791 | 0.881 | 15.0% |
| DCN-V2 | 0.651 | 35.1% | 46,909 | 0.927 | 12.0% |
| Polynomial Regression | 0.628 | 34.3% | 46,208 | 0.892 | 14.3% |
| Random Forest | 0.599 | 34.4% | 49,689 | 0.921 | 12.4% |
| TabNet | 0.551 | 38.2% | 52,729 | 0.922 | 12.4% |
| Kernel Regression | 0.497 | 39.0% | 56,765 | 0.890 | 14.7% |
| KNN | 0.496 | 39.2% | 56,756 | 0.889 | 14.8% |
| Decision Tree | 0.465 | 38.0% | 57,822 | 0.897 | 14.2% |
| Radius Neighbours | 0.433 | 41.0% | 59,099 | 0.870 | 15.9% |

**Types of mistakes** (`results/b3_external/b1_tuned_<family>_errors.png`; numbers in `b1_tuned_<model>_errors.csv`). For
each group of cars the graph shows the **bias**, the median of (predicted − actual) / actual, i.e. the *direction* of the
mistake (right of 0: the model asks too much), and the **MAPE**, the *size* of the mistake. The same pattern appears in every
family, so it comes from the data, not from one model (MLP numbers):
- **Age is the main mistake.** Cars up to 3 years old are priced almost right (bias +0.4%, MAPE 16%); 13–20-year-old cars
  are priced about 58% too high (MAPE 78%). In Morocco old cars keep their value, in Germany they lose it fast; the model
  learned the Moroccan depreciation.
- **Diesels are priced about 29% too high** (MAPE 47%, against 27% for petrol and 19% for hybrids): diesel is the most
  wanted fuel in Morocco (79% of the training cars) and less wanted in Europe. Electric cars are the one group priced
  slightly too low (−7%).
- **Brands:** German premium brands (BMW +30%, Land Rover +31%, Volkswagen +25%, Mercedes +23%) are over-priced most;
  Toyota, Skoda and Nissan are close (+3 to +6%). Most of that is the age and diesel effect again (many older German diesels).
- **Fiscal power:** filled from the Moroccan data, so its groups mostly mirror fuel and car type (7–8 CV, typical of
  diesels, has the largest bias).
- **The models still work on real-world cars from another market:** the best (MLP) keeps R2 0.71 against 0.93 at home,
  with MAPE 32% against 12%. The loss is the expected cost of another market and missing features. The MLP and Gradient
  Boosting transfer best; the neighbour-based models and the single tree worst (they copy the prices of similar Moroccan cars).
- **How to reduce the mistakes:** retrain or fine-tune on some local cars when moving to another market (the age and diesel
  effects are market-specific), add the missing features (Condition, Doors) to the new data, and refresh the model with
  newer listings because asking prices change over time.
