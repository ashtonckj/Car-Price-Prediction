# What the report must say

This file lists the points the written report **has to** explain, with the reasoning and the numbers behind them.
Smaller details (file layout, plot styling, metrics we dropped) are left out on purpose. Background and older measurements
are in the appendix at the end. Numbers are from 2026-11-19; re-run a family before quoting it if its code or data changed
(the complexity-curve numbers in section 6 are from 2026-11-20).

---

## 1. Which rows are used (data cleaning, before the split)

Applied to **every row before the train/test split** (`src/preprocessing/full.py`; `minimal.py` keeps the same rows,
so both Part A stages start from the same rows):

| Rule | Rows removed | Reason to give |
|---|---|---|
| No price | 25,483 | A target value is never invented |
| Price below 5,000 MAD | 557 | Not real asking prices. Even between 3,000 and 5,000 MAD, 98% of listings are under a fifth of the usual price of their model while being listed in good condition |
| Price above 2,000,000 MAD | 211 | Mostly typos and placeholder values; the few real ones are too rare to learn from |
| Placeholder prices (one digit repeated, e.g. 11111, 999999, 1111111) | 12 | Example: a 2011 Dacia Logan at 999,999 MAD while that model's median is 85,000. 9,999 and 99,999 are kept: they are normal "just under 10k / 100k" prices close to their model's median |
| **Price far from similar cars** (`drop_price_outliers`) | 1,027 | See below |
| Missing Brand, Model or Condition; duplicate listings; one corrupted row | the rest | Cannot be used / would be counted twice |

101,896 raw listings → **61,354 rows**.

**The "price far from similar cars" rule** must be explained properly, because it is the one rule that is not fixed in advance:

- **What it does.** Older cars are cheaper, so first the general year trend is taken out of the log price. Each car is then
  compared with the median of its Brand + Model (or its Brand, if the model has fewer than 5 listings). A car is removed if its
  price is more than 1.5 away in log price, i.e. about **4.5 times** too high or too low for that kind of car.
- **Why.** It catches wrong prices that a fixed minimum or maximum cannot see, e.g. a 2021 Dacia Sandero at 1,550,000 MAD or
  a 2011 Volkswagen Touareg at 3,000 MAD.
- **Say openly that this is a mild form of leakage.** The medians and the year trend are calculated from all rows, test rows
  included, and the rule can remove a test car *because of its price*, which could not be done for a new car whose price is
  unknown. Our lecturer agreed this is acceptable for this assignment. A leakage-free version (learn the medians and the
  trend from the training rows only, and filter only the training rows) is planned as an extra Part B variant if time allows.

---

## 2. The price range: the model is for cars between about 21,000 and 445,000 MAD

After the split, only cars whose price lies between the **2nd and 98th percentile of the training rows** are kept, in the
training, validation **and** test sets alike (`PRICE_QUANTILES` in `src/shared/core_utils.py`). On the final split that is
about 21,000–445,000 MAD. This is done in `a2_preprocessed` and `b1_tuned`, not in the baseline (section 3).

What to explain:

- **It is a scope, not cleaning.** It removes real cars at both ends (mostly luxury brands such as Mercedes-Benz, Audi, Land
  Rover and Porsche at the top, and very old cars at the bottom; 440 of the 12,271 test cars, 3.6%). Say plainly that the
  model is built for this price range and is not meant for luxury or scrap-value cars.
- **Why it is needed.** R2 and RMSE are calculated from squared errors in MAD, so a handful of very expensive cars dominate
  them. In an earlier version without a range, 50 test cars out of 12,464 (0.4%) made up 70% of all squared error: for
  example a car listed at 2,900,000 MAD was predicted at about 300,000.
- **Why q2–q98 and not q1–q99.** Both were tested; q2–q98 gave about +0.06 R2 more. The price for that is a narrower range of cars.
- **Why it is not leakage.** The two limits are learned from the training rows only (in cross-validation, from each fold's
  own training rows) and then applied to the other sets, just like a scaler. Test prices never set them.
- **Why test rows are filtered too.** The model is trained and scored on the same range of cars. (An earlier version
  trimmed only the training rows: the model could then never predict above about 650k while the test set still contained
  cars up to 2.9M, and it scored *worse*.)

---

## 3. Baseline (a1) vs full preprocessing (a2)

The baseline does the minimum: it starts from the same rows (section 1) but keeps **every** price after the split (no
q2–q98 range), fills missing numbers with 0 and missing text with "missing", one-hot encodes every raw column and predicts
the raw price. The full stage adds the price range, cleans the car names, converts mileage ranges to numbers, imputes missing
values from similar cars, scales the numbers and predicts the log price.

| Model | a1 CV R2 | a2 CV R2 | a1 test MAE | a2 test MAE | a1 MAPE | a2 MAPE |
|---|---|---|---|---|---|---|
| Ridge | 0.786 | 0.872 | 28,134 | 16,906 | 28.5% | 15.0% |
| Polynomial Regression | 0.785 | 0.886 | 28,165 | 15,851 | 28.5% | 14.0% |
| Random Forest | 0.834 | 0.913 | 19,370 | 13,714 | 16.6% | 12.4% |
| KNN | 0.756 | 0.863 | 27,168 | 17,648 | 26.5% | 15.9% |
| Radius Neighbours | 0.005 | 0.710 | 72,544 | 24,656 | 97.5% | 25.3% |
| MLP | 0.880 | 0.920 | 17,421 | 13,548 | 15.6% | 12.2% |
| DCN-V2 | 0.887 | 0.917 | 17,706 | 13,443 | 16.0% | 12.0% |

What to explain:

- **The dataset is feasible:** with minimal preprocessing most models already explain 72–89% of the price variance.
- **The two stages are not scored on the same cars** (a1: 12,271 test cars, a2: 11,831), because removing the cheapest and
  most expensive cars is itself part of the full preprocessing. So the a1 → a2 gain has two causes that must both be named:
  (1) the price range removes the cars that are hardest to predict, and (2) the column preprocessing (names, imputation,
  scaling, log target). An earlier measurement with both stages on exactly the same cars showed that the column
  preprocessing alone still lowers the typical error by about a quarter (MAE about 26k → 19–21k MAD for the linear models),
  while R2 rose only 0.02–0.05.
- **Radius Neighbours in a1 (R2 0.005) is not a bug, and it shows why scaling matters.** It averages the cars within a
  fixed distance of 1.0. Without scaling, Year and Mileage are in years and kilometres, so almost no car has another car that
  close, and it falls back to predicting the average price for nearly every car, which by definition gives R2 ≈ 0. With
  scaled features (a2) the same model reaches 0.71.

---

## 4. Target: log price, converted back with a smearing correction

- **Why log price:** prices are strongly right-skewed and errors grow with the price, so the full stage predicts
  log(1 + price). This is better than predicting the price directly: with raw Price as the target, test R2 fell for
  every linear model (Ridge 0.541 → 0.495, Polynomial 0.591 → 0.556; measured on an earlier version of the data).
- **All metrics are on the price in MAD**, never on the log price, so every stage and model is directly comparable.
- **Smearing (Duan, 1983).** Converting a log prediction back with `exp` gives roughly the *median* price for those
  features, which on average underestimates the price. Every predicted price is multiplied by
  `mean(exp(residual))`, where the residuals are the log-price errors on the **validation** rows (never the test rows).
  It raised R2 on every CV fold (Ridge +0.011 to +0.016, Polynomial +0.001 to +0.010; measured on an earlier version of the data)
  and makes little difference for Random Forest. It does nothing for the baseline, which predicts the price directly.

---

## 5. How leakage is handled (summary paragraph)

- The data is split **70% train / 10% validation / 20% test**. The test set is scored once at the end.
- Imputation values, one-hot categories, scalers and the q2–q98 limits are learned from the training rows only; the
  smearing factor from the validation rows.
- Model selection and tuning use the validation rows or 5-fold cross-validation on the 70% train part, never the test set.
- Before the split, fixed rules (section 1) are applied identically to all rows. **The one exception** is the "price far
  from similar cars" rule, which uses statistics of all rows: state it as a known, accepted limitation (section 1).

---

## 6. Overfitting check (complexity curves)

`src/analysis/a2_preprocessed_complexity_curve.py` → `results/a2_preprocessed/complexity_curve.png` (default models) and
`src/analysis/b1_tuned_complexity_curve.py` → `results/b1_tuned/complexity_curve.png` (tuned models). For one model per family
(Polynomial Regression: degree, Random Forest: max_depth, KNN: k, MLP: training epochs), one setting that controls
complexity is swept while the others stay at their default values (a2) or their tuned values (b1), and the error on the
training rows is compared with the error on the validation rows. The test rows are not used here; they are only used for
the `*_pred_vs_actual.png` graphs and the test scores. In b1 a grey dashed line marks the value the tuning picked.

- **Overfitting shows as** a training error that keeps falling while the validation error stops falling or rises again.
  Examples to point at: polynomial degree 4 (validation RMSE jumps from 26.1k to 29.0k MAD while training error keeps
  falling) and KNN with k = 1 (training error about 5k, validation 34k: every car is its own nearest neighbour).
- **Random Forest depth:** the curve shows whether the tuned depth is sensible. With depth fixed at 15 (the first RF
  grid), validation RMSE was about 25.0k MAD, against 22.1k from depth 45 up, so depth 15 was too shallow; the tuned
  depth is now 50. Beyond about depth 45 nothing changes, because the trees are already fully grown.
- **Neural networks (MLP):** trained for 100 epochs without early stopping, the training error keeps falling while the
  validation error flattens and wobbles, which is what early stopping (patience 10, the weights of the best epoch are
  kept) cuts off. Quote the epoch of the lowest validation error from the figure after re-running it.
- **Which neural network:** we carry the **MLP** forward rather than DCN-V2. On cross-validation the MLP is ahead at both
  stages (a2: 0.920 vs 0.917; tuned: 0.920 vs 0.919); DCN-V2 was ahead on one test split by 0.001. The differences are
  smaller than the CV standard deviation (0.005–0.009), so they are a tie, and the simpler model is preferred.
- **Be honest about tuning gains:** tuning did not change the MLP's CV R2 (0.920 → 0.920); its test gain (0.921 → 0.930) is
  within the noise of a single split.

---

## 7. How to read the results

- **Report CV mean ± std next to the test score.** The test score depends on which cars happen to be in the 20%:
  changing the data by even a few rows reshuffles the split and can move test R2 by about ±0.03. Preprocessing decisions
  were made on CV R2 for this reason, and the CV standard deviation is the evidence that a result is stable.
- **Report MAE and MAPE next to R2:** R2 and RMSE are dominated by the most expensive cars.

---

## 8. Limitations to state

- **Price range:** only cars between about 21k and 445k MAD; luxury and very cheap cars are out of scope.
- **The "price far from similar cars" rule** uses all rows, test rows included (section 1).
- **The neural networks** also use the validation rows for early stopping, so for them the smearing factor is learned on
  rows that already influenced training. The effect is small, but say so in one sentence.
- **The data is from 2024 listings**; asking prices drift over time, which matters for the new-data test in Part B.
- **New-data test (section 10):** the external cars have no Condition or Doors (filled with training values), their fiscal power
  and equipment are approximated, and only car models the model was trained on, priced inside its training price range, are
  tested: the model makes no claim for other cars.
- **Asking prices, not sale prices:** listings contain negotiation margin and some wrong values that no rule can catch.

---

## 9. Ensembles (Part B step 2, `b2_ensemble`)

`src/ensemble/` combines tuned b1 models; results go to `results/b2_ensemble/` (numbers to be added after the rerun).
Three combinations: Polynomial + Gradient Boosting + Kernel Regression + MLP (one model per family, the MLP being the neural network carried forward), Random Forest + Gradient Boosting,
and MLP + DCN-V2. Each is combined in two ways:

- **Stacking:** each base model is trained 5 times on 4/5 of the training rows and predicts the remaining fifth, so every
  training car gets a prediction from a model that did not see it ("out-of-fold"). A Ridge regression learns how to weigh the
  base models from those predictions.
- **Blending:** the same out-of-fold predictions, but the weights are found directly (between 0 and 1, adding up to 1) to give
  the lowest RMSE. The weights are easy to explain: they say how much each model counts.

What to explain: the ensembles are compared with the best single b1 model, not in the a1 / a2 / b1 stage comparison (they are
not the same models under a new setting). An ensemble helps most when its models make *different* mistakes, so the
one-model-per-family combination is expected to gain more than RF + GB or MLP + DCN-V2, which are similar to each other.
State the extra cost: every base model is trained 5 times. The networks stop early on the 10% validation rows, never on the out-of-fold
rows, so the out-of-fold predictions the weights are learned from stay honest.

---

## 10. Do the saved models work on new data? (Part B step 3, `b3_external`)

The spec asks to "measure how well the model performs on new data" and to "look at the types of mistakes it makes". We test the
saved models on real-world used-car listings from another market: KTHESAT, "European Used Cars 100k Sample", Hugging Face,
CC BY-NC 4.0 (100,000 mobile.de / AutoScout24 listings, mostly Germany). The repository is public, so
scraped website listings are not used.

**Nothing is retrained.** `src/external/evaluate_saved_models.py` loads each model saved from `cars_full.csv` (with the
imputation, encoders, scaler, target scaler and smearing learned from the Moroccan training rows) and predicts the listings.
`src/preprocessing/external_europe.py` gives them the same columns as `cars_full.csv`:
- **Price** converted to MAD at one fixed rate (1 EUR = 10.77 MAD, the 2026 average) and rounded to 10 MAD; the same price rules
  as `full.py`. One rate for every car does not change R2 or MAPE.
- **Car names** written as in the Moroccan data (BMW 320 -> serie 3, Mercedes C 220 -> classe c, MINI -> Mini ...).
- **Equipment Amount:** how many of the 16 Moroccan equipment items the listing's (German) equipment list contains.
- **Mileage** in km put into the training data's ranges, then the range midpoint, as in `full.py`.
- **Fiscal Power (CV):** the listings give horsepower, while the Moroccan CV is based on engine size and fuel, so it is filled
  like any missing CV: the Moroccan CV of the same Brand + Model + Fuel.
- **Condition and Number of Doors** are not in the dataset: most common training condition, training Brand/Model doors.

**Which cars.** The test uses the cars the saved models are made for: real-world cars of the car models they were trained on
(Brand + Model pairs with at least 10 cars in the Moroccan training rows, the encoder's minimum, so each has its own column in
the model), priced inside the training price range (q2-q98 of the Moroccan training prices, about 21k-445k MAD; the same rule
the Moroccan test set gets in a2 / b1, so the two scores are comparable). Both rules come from the training rows only, never
from the European prices or the errors. That leaves 45,995 listings.

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

**Types of mistakes** (`results/b3_external/b1_tuned_<family>_errors.png`; numbers in `b1_tuned_<model>_errors.csv`). For each group
of cars the graph shows the **bias**, the median of (predicted - actual) / actual, so it tells the *direction* of the mistake
(right of 0: the model asks too much, left: too little), and the **MAPE**, the *size* of the mistake. The same pattern appears
in every family, so it comes from the data, not from one model (MLP numbers):
- **Age is the main mistake.** Cars up to 3 years old are priced almost right (bias +0.4%, MAPE 16%); 13-20-year-old cars are
  priced about 58% too high (MAPE 78%). In Morocco old cars keep their value, in Germany they lose it fast; the model learned the
  Moroccan depreciation.
- **Diesels are priced about 29% too high** (MAPE 47%, against 27% for petrol and 19% for hybrids): diesel is the most wanted fuel
  in Morocco (79% of the training cars) and less wanted in Europe. Electric cars are the one group priced slightly too low (-7%).
- **Brands:** German premium brands (BMW +30%, Land Rover +31%, Volkswagen +25%, Mercedes +23%) are over-priced most; Toyota,
  Skoda and Nissan are close (+3 to +6%). Most of that is the age and diesel effect again (many older German diesels).
- **Fiscal power:** filled from the Moroccan data, so its groups mostly mirror fuel and car type (7-8 CV, typical of diesels,
  has the largest bias).
- **The models still work on real-world cars from another market:** the best (MLP) keeps R2 0.71 against 0.93 at home, with
  MAPE 32% against 12%. The loss is the expected cost of another market and missing features (no Condition or Doors; fiscal
  power and equipment approximated). The MLP and Gradient Boosting transfer best; the neighbour-based models and the single tree
  worst (they copy the prices of similar Moroccan cars).
- **How to reduce the mistakes:** retrain or fine-tune on some local cars when moving to another market (the age and diesel
  effects are market-specific), add the missing features (Condition, Doors) to the new data, and refresh the model with newer
  listings because asking prices change over time.

---

## Appendix: earlier preprocessing measurements (2026-11-17)

> **Superseded (2026-11-19):** the measurements below are from 2026-11-17 (q1–q99 in every stage, no per-model filter). The team then chose
> q2–q98 (a2/b1 only) and kept the per-model filter `drop_price_outliers` (the lecturer accepted its mild leakage). The sections above describe the
> current choices; this section is kept as the record of what was measured and why.

All numbers are for `a2_preprocessed`, R2 on the price in MAD. **Preprocessing choices are compared on the 5-fold CV R2**, not on the test R2:
removing even a few rows reshuffles the whole 70/10/20 split, so the test set becomes different cars and its R2 moves by about ±0.03 from that alone.

### Price rules (in `full.py`, applied to every row before the split)

| Rule | Rows removed | Why |
|---|---|---|
| No price | all of them | A target is never invented |
| Price below **5,000** MAD | 47 (3k–5k) | 98% of them are under 1/5 of their model's median price while mostly listed in good condition: wrong prices, not real cars. 10k was also tried: no extra gain, and the 5k–10k range is more mixed (some for-parts / damaged cars) |
| Price above **2,000,000** MAD | ~30 (2M–3M) | 5-fold CV R2 +0.08 (Ridge 0.527 → 0.611, Polynomial 0.557 → 0.637). The model could not predict these anyway (e.g. 2.9M predicted as 300k). Scope of the model: cars up to 2M MAD. 1.5M gave +0.04 more but drops more of the high range |
| Placeholder prices (`is_placeholder_price`) | 12 | One digit repeated 4+ times: 999,999 for a 2011 Dacia Logan (model median 85k), 1,111,111 for a 2005 Honda City (median 35k). 9,999 and 99,999 are kept: they sit near their model's median ("just under 100k" pricing) |
| Duplicates | — | `drop_duplicates()` |

Result: 62,232 rows.

### Price scope q1–q99 (after the split, every stage; now `PRICE_QUANTILES` = q2–q98 in `core_utils.py`, a2/b1 only)

The limits are the 1% and 99% quantiles of the **training rows** (15,000–630,000 MAD), applied to train, val and test alike; in CV each
fold learns them from its own training rows. Computed on the whole dataset instead they are almost the same, but learning them after the split
costs nothing and keeps the test rows out of the decision. The limits are real training prices, so a1 (Price) and a2 (Log Price) keep the same rows.

This is a **scope**, not cleaning: 622 cars above the limit (Mercedes-Benz 120, Audi 97, Land Rover 70, Volkswagen 55, Porsche 51, BMW 37 ...)
and 507 below it (median year 2000) are real cars that the model is not meant for. State it in the report:
*"the model prices cars between 15k and 630k MAD"*. Polynomial CV R2 0.636 (5k–2M only) → 0.820 (q1–q99); q2–q98 (20k–480k) gave 0.854 but drops twice as many cars.

Not the same as the q1–q99 trimming that was removed (next table): that one trimmed only the training rows, so the test set kept cars the
model had never seen the like of. Here every set covers the same range.

### Tried and not used

| Idea | Result | Why not |
|---|---|---|
| Trim **only the training rows** to the q1–q99 price range after the split | Test R2 0.591 vs **0.614** without it (Polynomial) | Only training rows were trimmed, so the model could never predict above ~650k while the test set went up to 2.9M. Removed |
| Raw `Price` as target instead of `Log Price` | Polynomial test R2 0.556 vs 0.591, Ridge 0.495 vs 0.541 (CV agrees); Random Forest about equal | Log Price kept |
| `sqrt(Price)` as target | Between the two | Log Price kept |
| Per-model outlier filter (drop a price more than 3× or 5× away from its Brand+Model median) | On training rows only: CV R2 change 0.000 to −0.007. Also on test rows: 0.82 | The model does not get better; the gain comes only from deleting ~4% of test cars **because of their actual price**, which cannot be done for a new car whose price is unknown |

### Smearing (log-price bias correction)

A model trained on log price predicts something close to the median price, so `expm1(prediction)` underestimates the average price.
Duan's smearing multiplies every predicted price by `mean(exp(log residuals))`. The factor is learned on the **validation** rows (never on test), in
`split.smear()` (`core_utils.py`), and is applied in every model script. For `a1_baseline` it does nothing. Same splits, without → with smearing:
Ridge test R2 0.517 → 0.530 (every CV fold +0.011 to +0.016), Polynomial 0.552 → 0.557 (+0.001 to +0.010). For Random Forest it makes almost no difference.
The neural networks also use the validation rows for early stopping, so for them the factor is learned on rows they have already seen; the effect is small.

### What may be done to the test rows

Leakage means information from the test rows reaching the model or a decision about it. Removing a test row is not leakage in itself; the
question is whether it changes **which cars the score describes**.

Allowed:
- **Fixed rules decided without looking at the test score, applied to every row the same way, before the split**: no price, impossible or
  placeholder prices, duplicates, corrupted rows, the 5k–2M scope. These are data-quality rules or a stated scope ("cars up to 2M MAD"), and they would be applied to new data too.
- **Transformations and limits learned on training rows, then applied to test rows**: imputation, encoding, scaling, the q1–q99 price scope,
  the smearing factor (validation rows). The price range is a stated scope: one range for every car, which you can tell a user up front.

Not allowed:
- Anything **fitted on** test rows (a scaler, a median, a quantile limit).
- Choosing a rule or threshold **because it raises the test score**. Choose on CV R2 (training rows) and give a data-quality reason.
- Removing test rows **because the model gets them wrong**, or because their price is unusual for that car (the per-model filter above). The score would then describe only "normally priced" listings, not real ones.

A test set that still contains rare but real cars does **not** give a "false" low R2: that is how hard the real data is. Only wrong values
(typos, placeholders) make it unfairly low, and those are removed by the rules above for every row. Report MAE and MAPE next to R2: R2 and RMSE
are dominated by the few most expensive cars.

### Linear results at the time (2026-11-17: 5k–2M, placeholders removed, q1–q99 scope, smearing)

Both stages: same rows, train 42,784 / val 6,109 / test 12,210.

| Model | a1_baseline CV R2 | a1 test R2 / MAE / MAPE | a2_preprocessed CV R2 | a2 test R2 / MAE / MAPE |
|---|---|---|---|---|
| Ridge | 0.773 ± 0.008 | 0.787 / 26,216 / 28.3% | 0.797 ± 0.007 | 0.806 / 21,083 / 19.4% |
| Huber | 0.741 ± 0.007 | 0.752 / 24,707 / 23.2% | 0.799 ± 0.006 | 0.810 / 20,736 / 19.3% |
| Polynomial Regression | 0.772 ± 0.009 | 0.786 / 26,240 / 28.3% | 0.820 ± 0.009 | 0.837 / 19,334 / 17.9% |

With the same rows, the baseline is already strong in R2; the full preprocessing mainly shows in MAE and MAPE (about a quarter lower error).
