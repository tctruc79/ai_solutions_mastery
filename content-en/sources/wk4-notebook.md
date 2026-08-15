---
type: source
title: "WK4 Notebook — Model Selection, AI Explainability, AI Fairness, Deployment"
tags: [model-selection, pca, tsne, kmeans, dbscan, anomaly-detection, shap, fairlearn, deployment, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk4-model selection-ai explainability-Deployment-updatedv2.ipynb"
---

## Source Info

File `honeyg_ai-practitioner-wk4-model selection-ai explainability-Deployment-updatedv2.ipynb`,
197 cells — the LONGEST notebook across the 4 weeks. Libraries NEW versus Week 3: `shap`,
`streamlit`, `joblib`, `fairlearn` (`MetricFrame`, `ExponentiatedGradient`, `DemographicParity`,
`EqualizedOdds`, `TruePositiveRateParity`, `ThresholdOptimizer`), plus from `sklearn`:
`sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans, DBSCAN}`,
`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`,
`sklearn.utils.class_weight.compute_sample_weight`, `sklearn.metrics.ConfusionMatrixDisplay`.

## Learning Objectives

The notebook splits into 6 major parts, moving from "pick the best model" to "ship it":
**Modelling Part I–III** (recaps LR/RF/9-model/XGBoost/imbalance/Deep Learning from WK3, then
expands into a larger grid) → **Modelling Part IV** (unsupervised learning: PCA, t-SNE, K-Means,
DBSCAN, anomaly detection) → **AI Explainability** (SHAP) → **AI Fairness** (FairLearn) →
**Deployment Pipeline** (reruns everything from scratch via packaged functions, compares a
**54-combination grid** of models × imbalance strategies, re-checks on the real TEST SET, retunes
with Hyperopt, saves the model via `joblib` for a separate Streamlit app).

## Notebook Structure

- **Feature Engineering** — identical to WK2/WK3.
- **Modelling Part I** — recaps LR + RF (same results as WK2/WK3).
- **Modelling Part II** → "Comparison of 9 Different Models" (2 code versions: "Previous
  Code"/"Updated Code" — the latter adds XGBoost to the table) → "XGBoost" (train/validate +
  `gain` feature importance) → "XGBoost for Imbalanced Data" (SMOTE beats ADASYN/Undersample,
  Hyperopt tuning) → "Deep Learning with Tensorflow/Keras" (base model already FIXES WK3's `tanh`
  bug — uses `sigmoid`/`binary_crossentropy` from the start — then compares Base vs CTGAN
  Oversampling vs Undersampling for BOTH Deep Learning AND XGBoost).
- **Modelling Part III** → "Finding Best Model: 45 Supervised Model + Imbalanced Data Solutions"
  — an EARLIER draft of the comparison grid (only 4 strategies: SMOTE/ADASYN/UNDERSAMPLE/ORIGINAL,
  9 models = 36 actual combinations despite the heading saying "45") — SUPERSEDED by the fuller
  grid in the Deployment Pipeline section below.
- **Modelling Part IV** → "Dimensionality Reduction with PCA" (95% variance → 10 components) →
  "Dimensionality Reduction/Visualisation with TSNE" (with/without PCA pre-processing) → "K-Means
  Clustering" (elbow method) → "Clustering with DBSCAN" → "Unsupervised Anomaly Detection"
  (Isolation Forest, LOF, Autoencoder).
- **AI Explainability** → "Global Explainability" (SHAP `summary_plot` beeswarm + bar) → "Local
  Explainability" (SHAP `waterfall` for individual rows).
- **AI Fairness** → "FairLearn" (`MetricFrame` measured by group, `ExponentiatedGradient` with 3
  constraints, `ThresholdOptimizer`, manual per-group threshold adjustment).
- **Deployment Pipeline** → reruns the ENTIRE pipeline via packaged functions
  (`generate_train_test_data`, `build_features`...) → "Modeling, Model Selection and Model
  Fine-Tunning" (**54-combination grid** = 9 models × 6 strategies: SMOTE/ADASYN/UNDERSAMPLE/PCA
  Features/CTGANS Oversampling/ORIGINAL) → "Selecting the Best Model - Testing on Test Set"
  (re-checks on the real TEST SET) → "Hyperparameter Tunning for Best Model" (second Hyperopt
  pass) → "Saving the Best Model" (`joblib.dump`) → rebuilds SHAP + FairLearn on the final model
  → saves `Test_X_transformed`, `Test_y_transformed`, `feature_names` to `.npy` files for a
  separate Streamlit app.

## Libraries & Tools

`shap` (`TreeExplainer`, `summary_plot`, `plots.waterfall`), `fairlearn.metrics.MetricFrame`,
`fairlearn.reductions.{ExponentiatedGradient, DemographicParity, EqualizedOdds,
TruePositiveRateParity}`, `fairlearn.postprocessing.ThresholdOptimizer`,
`sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans, DBSCAN}`,
`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `joblib`,
`streamlit`.

## Technical Walkthrough

### The 54-combination grid — 9 models × 6 imbalance strategies

The "Deployment Pipeline" section reruns the full comparison with 6 strategies: SMOTE, ADASYN,
Undersampling, **PCA Features** (9 components), **CTGAN Oversampling**, and the original data
(ORIGINAL) — for EACH of 9 models → 54 combinations. Result (the `model_list` table, sorted by
descending AUPRC): **XGBoost + CTGAN (0.9639)** leads, closely followed by Gradient Boosting +
CTGAN (0.9634), then XGBoost + SMOTE (0.9590), XGBoost + ADASYN (0.9507).

### Confirmation on the real TEST SET — the turning point of the whole course

The "Selecting the Best Model - Testing on Test Set" section trains XGBoost on the ORIGINAL data
(no resampling) then predicts on `Test_X_transformed` (the REAL test set, never resampled) →
**AUC-PR = 0.8024**. Repeated with XGBoost trained on CTGAN-resampled data → predicting on that
SAME real test set → **AUC-PR = 0.7951** — LOWER than the non-resampled version, despite CTGAN
having just "won" with a 0.9639 cross-validation AUPRC! This is exactly the lesson foreshadowed
by [[auprc]] and [[xgboost]] from [[wk3-modelling-imbalance-deep-learning]]: AUPRC measured on
resampled data (even via CV) can be falsely optimistic, since a synthetic sample may be "too
similar" to the real point it was generated from — always confirm against a REAL test set that
was never resampled.

### PCA, t-SNE, K-Means, DBSCAN, anomaly detection

PCA: 95% variance needs 10/19 components (yet the 54-combination grid uses 9 components — not
fully matching the 10 just found). t-SNE: non-linear dimensionality reduction for 2D
visualisation, tried on both the raw data and after PCA. K-Means: uses the elbow method then
picks `optimal_k = 10` — see [[kmeans-dbscan-clustering]]. DBSCAN: `eps=0.5, min_samples=40`,
drops noise points (label -1) before computing per-cluster means. Anomaly Detection: Isolation
Forest (AUPRC-like 0.6161), LOF (0.6082), a hand-written Autoencoder (0.5753) — see
[[anomaly-detection]].

### SHAP — global and local explanations

`shap.TreeExplainer(final_model)` on the final XGBoost model → `shap.summary_plot` (beeswarm:
every data point, every feature, simultaneously — global) and `shap.plots.waterfall` (one
specific data point, how much each feature contributed to that prediction — local) — see [[shap]].

### FairLearn — measuring and mitigating bias against sensitive groups

`MetricFrame` measures accuracy/precision/AUPRC/TPR SEPARATELY per group (`is_male`,
`is_married`, `race_white/black/asian-pac-islander`). Three mitigation approaches:
`ExponentiatedGradient` (in-processing — constrains during training itself, tried with all 3
constraints `DemographicParity`/`EqualizedOdds`/`TruePositiveRateParity`), `ThresholdOptimizer`
(post-processing — adjusts the classification threshold after the model already exists, per
`demographic_parity`), and MANUAL per-group threshold adjustment (0.3 for the not-married group,
0.7 for the married group) — see [[fairlearn]].

## Key Numbers

| Model / technique | AUPRC (CV) | AUPRC (real test set) |
|---|---|---|
| XGBoost + CTGAN Oversampling (wins the 54-combination grid) | 0.9639 | 0.7951 |
| Gradient Boosting + CTGAN Oversampling | 0.9634 | — |
| XGBoost + SMOTE | 0.9590 | — |
| XGBoost (original data, no resampling) | 0.7976 | **0.8024** |
| XGBoost + Hyperopt tuning (on CTGAN, second pass) | 0.9653 | not re-checked in the notebook |
| Isolation Forest (anomaly detection) | AUPRC-like 0.6161 | — |
| LOF (anomaly detection) | AUPRC-like 0.6082 | — |
| Hand-written Autoencoder (anomaly detection) | AUPRC-like 0.5753 | — |
| Optimal PCA components (95% variance) | 10/19 | (the 54-combination grid uses 9 instead) |
| Optimal K-Means k (chosen manually) | 10 | (the code comment says "3") |

## Quirks / Issues Found

Three genuine technical issues, not intentional illustration:

1. **The "best model" is picked by comparing CHARACTERS, not AUPRC values** — the line
   `best_model = max(results, key=lambda x: x[1])` iterates over the dict `results`' KEYS
   (model-name strings, e.g. `"XGBoost SMOTE"`) rather than (key, value) pairs; `x[1]` is
   therefore the 2nd CHARACTER of the model name (e.g. `'G'` in `"XGBoost..."`), not an AUPRC
   score — the printed result is entirely wrong: **"Best model based on AUPRC: Extra Trees SMOTE
   with an AUPRC of 0.9026"**, while the `model_list` table (correctly sorted via
   `.sort_values()`) right below it shows the REAL winner is **XGBoost + CTGAN Oversampling
   (0.9639)**. Correct fix: `max(results.items(), key=lambda kv: kv[1][0])`.
2. **The saved/deployed model is never re-checked on the test set AFTER tuning** — the "Testing
   on Test Set" section only checks 2 UNTUNED models (original: 0.8024, CTGAN: 0.7951); Hyperopt
   then further tunes on the CTGAN data (reaching 0.9653 CV AUPRC), and this model is
   `joblib.dump`ed as the final deployed version — but the notebook never goes back to measure
   THIS tuned model's AUPRC on the real test set. Since the section right above just demonstrated
   that high CV doesn't guarantee high test performance, this is a notable evaluation gap.
3. **K-Means elbow method: code doesn't match the comment** — a cell comment reads "Choose k from
   the elbow plot visually, let's say it appears to be 3" but the very next line sets
   `optimal_k = 10` — unclear whether this is a typo or a changed decision with a stale comment,
   but it illustrates a practical pitfall: always cross-check comments against the code that
   actually runs, never trust a comment blindly.

## Wiki Links

[[wk4-model-selection-explainability-fairness-deployment]] · [[wk3-notebook]] · [[xgboost]] ·
[[auprc]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[pca-tsne-dimensionality-reduction]] · [[kmeans-dbscan-clustering]] · [[anomaly-detection]] ·
[[shap]] · [[fairlearn]] · [[overview]]
