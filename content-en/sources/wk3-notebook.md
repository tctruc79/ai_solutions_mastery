---
type: source
title: "WK3 Notebook — Modelling II (9 Models, XGBoost, Imbalance, Deep Learning)"
tags: [modelling, xgboost, imbalanced-data, deep-learning, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE - ADASync - CTGANs.ipynb"
---

## Source Info

File `honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE -
ADASync - CTGANs.ipynb`, 64 cells. Libraries NEW versus Week 2: `xgboost`, `tensorflow.keras`,
`ctgan.CTGAN`, `imblearn.over_sampling.{SMOTE, ADASYN}`, `imblearn.under_sampling.
RandomUnderSampler`, `hyperopt`, more `sklearn.ensemble` classes (GradientBoosting, AdaBoost,
ExtraTrees, Bagging), `sklearn.naive_bayes.GaussianNB`.

## Learning Objectives

The notebook splits into 2 major parts: **Feature Engineering** (reuses the WK2 pipeline
UNCHANGED) → **Modelling Part I** (recaps LR + RF from WK2) → **Modelling Part II** (compare 9
models via AUPRC, XGBoost deep-dive, class-imbalance handling, Hyperopt tuning, Deep Learning +
CTGAN).

## Notebook Structure

- **Feature Engineering** — identical to WK2 (`generate_train_test_data`,
  `CustomFeatureCreator`, `ColumnTransformer`, `Pipeline`).
- **Modelling Part I** — recaps LogisticRegression + RandomForest (results identical to WK2,
  same pipeline + same random_state).
- **Modelling Part II** → "Comparison of 9 Different Models" (Naive Bayes, LR, Decision Tree,
  RF, Gradient Boosting, XGBoost, Extra Trees, AdaBoost, BaggingClassifier — measured by AUPRC
  via 10-fold `StratifiedKFold`) → "XGBoost" (train + validate with a custom AUPRC eval metric,
  build the final model, feature importance by `gain`) → "XGBoost for Imbalanced Data" (compare
  SMOTE/ADASYN/Random Undersampling/original data, then Hyperopt tuning on the best strategy) →
  "Deep Learning with Tensorflow/Keras" (base model, then compare CTGAN oversampling vs
  undersampling vs no treatment).

## Libraries & Tools

`xgboost` (both the sklearn-style `XGBClassifier` API and the native `xgb.train`/`xgb.DMatrix`
API), `imbalanced-learn` (`SMOTE`, `ADASYN`, `RandomUnderSampler`), `hyperopt` (`fmin`,
`tpe.suggest`, `hp`, `Trials`), `tensorflow.keras` (`Sequential`, `Dense`, `Dropout`,
`EarlyStopping`), `ctgan` (`CTGAN` — Conditional Tabular GAN for synthetic data generation).

## Technical Walkthrough

### Comparing 9 models — switching entirely to AUPRC

Instead of accuracy/F1 like WK2, every model comparison from here on uses
`make_scorer(average_precision_score, needs_proba=True)` via `StratifiedKFold(n_splits=10)` —
see [[auprc]]. Result: XGBoost (0.7976) and Gradient Boosting (0.7962) lead, Decision Tree
(0.5069) and Extra Trees (0.6079) are weakest on the ORIGINAL data (before imbalance treatment).

### XGBoost — native API with a custom eval metric

Uses `xgb.DMatrix` + `xgb.train()` (not `XGBClassifier.fit()`) to pass a
`custom_metric=auprc_eval` (a hand-written function computing AUPRC every boosting round) and
`scale_pos_weight` (negative/positive class ratio) so the model self-balances somewhat even
before any resampling — see [[xgboost]].

### Handling imbalance — SMOTE vs ADASYN vs Undersampling

`SMOTE(random_state=42).fit_resample(X, y)`, `ADASYN(...)`, `RandomUnderSampler(...)` — each
yields a new, more balanced training set, re-evaluated via
`cross_val_score(scoring='average_precision')`. Result: **SMOTE wins clearly** (AUPRC ≈ 0.959
versus original data), so `best_X_train, best_y_train = Train_X_smote, Train_y_smote` is chosen
for the next tuning step — see [[imbalanced-data-resampling]].

### Hyperopt — finding hyperparameters via TPE instead of grid search

`fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50)` — the Tree-structured
Parzen Estimator (TPE) algorithm INTELLIGENTLY picks the next parameter set based on previous
trial results, instead of an even grid sweep (grid search) or pure randomness — see
[[hyperopt-tpe]]. Result: AUPRC rises from 0.959 (SMOTE, default params) to 0.9635 (SMOTE +
tuned).

### Deep Learning — the base model FAILS, then CTGAN oversampling

The first Keras model (`Dense(64)→Dropout→Dense(32)→Dropout→Dense(2, activation='tanh')` +
`categorical_crossentropy`) produces **loss = nan** from epoch 1 — a genuine technical bug in
the original notebook (incorrectly pairing `tanh` output with `categorical_crossentropy`). The
following section uses a more correct model (`sigmoid` + `binary_crossentropy`) to compare Base
Model vs **CTGAN Oversampling** (GAN-generated synthetic minority samples) vs Undersampling —
see [[ctgan]].

## Key Numbers

| Model / technique | AUPRC |
|---|---|
| XGBoost (original data, no imbalance treatment) | 0.7976 |
| Gradient Boosting (original data) | 0.7962 |
| Decision Tree (original data, weakest) | 0.5069 |
| XGBoost + SMOTE | 0.9590 |
| XGBoost + ADASYN | ~0.949 |
| XGBoost + SMOTE + Hyperopt tuning | 0.9635 |
| Base Deep Learning (`tanh`+`categorical_crossentropy` bug) | loss = NaN |
| Deep Learning + CTGAN oversampling (1 illustrative fold) | AUPRC ≈ 0.728 |

## Quirks / Issues Found

The first Deep Learning model in the "Build Base Deep Learning Model" section produces `loss:
nan` throughout all 20 epochs — a genuine technical bug (not intentional illustration): the
output layer uses `activation='tanh'` (range [-1,1]) but the `categorical_crossentropy` loss
requires probabilities in [0,1] — this mismatch causes NaN from the very first round. The
notebook doesn't fix this cell, it simply switches to a different architecture
(`sigmoid`+`binary_crossentropy`) later — the lesson is carried into
[[wk3-modelling-imbalance-deep-learning]] as a practical pitfall.

## Wiki Links

[[wk3-modelling-imbalance-deep-learning]] · [[wk2-notebook]] · [[wk4-notebook]] · [[xgboost]] ·
[[auprc]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] · [[overview]]
