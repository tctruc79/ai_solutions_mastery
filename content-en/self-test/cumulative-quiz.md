---
type: self-test
title: "Cumulative Self-Test — Cross-Week Synthesis (WK1–WK4)"
tags: [self-test, cumulative, wk1, wk2, wk3, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

A LIVING document, now covering the FULL 4 weeks (WK1–WK4) of the course. It plays the role
DBC's `exam-prep.md` plays there, but this is a quick-reference + CROSS-WEEK comparison question
bank, not a replacement for each week's own quiz ([[wk1-quiz]], [[wk2-quiz]], [[wk3-quiz]],
[[wk4-quiz]]). Every comparison below deliberately starts from WK1 where possible — many points
that seem to belong only to WK2–WK4 actually have roots in WK1 (disguised missing data → data
leakage; fairness raised early → FairLearn addresses it).

## Quick Reference

| Week | Focus | Main Metric | New Technique | Key Pitfall |
|---|---|---|---|---|
| WK1 | [[wk1-data-understanding]] — CRISP-DM: Collect→Describe→Explore→Verify | descriptive/exploratory, no modelling yet | [[point-biserial-correlation]], [[network-graph-categorical-eda]] | disguised missing data (e.g. `?` not real NaN) |
| WK2 | [[wk2-data-preparation-feature-engineering]] — 4-step Data Prep + baseline model | accuracy, F1 (per-class) | [[sklearn-column-transformer-pipeline]], [[hierarchical-clustering-categorical-grouping]] | high accuracy (83-84%) is an illusion under class imbalance |
| WK3 | [[wk3-modelling-imbalance-deep-learning]] — 9 models, XGBoost, resampling, Deep Learning | AUPRC (official from here on) | [[imbalanced-data-resampling]] (SMOTE/ADASYN/Undersampling), [[hyperopt-tpe]], [[ctgan]] | resampling BEFORE the CV split → subtle data leakage; final-layer activation mismatched with loss → `nan` |
| WK4 | [[wk4-model-selection-explainability-fairness-deployment]] — 54-combination grid, PCA/t-SNE/K-Means/DBSCAN, SHAP, FairLearn, Deployment | AUPRC on CV **and** on the real test set (2 numbers, not 1) | [[pca-tsne-dimensionality-reduction]], [[kmeans-dbscan-clustering]], [[anomaly-detection]], [[shap]], [[fairlearn]] | high CV (0.9639) on resampled data ≠ high test score (0.7951) — the biggest lesson of the whole course |

## Cross-Week Comparisons

### 1. Why the metric changes so many times: accuracy → F1 (per-class) → AUPRC (CV) → AUPRC (real test set)

WK1 has no modelling yet — but lays the groundwork by correctly CLEANING the target (`>50K.` →
`>50K`), a prerequisite for every later metric to compute correctly. WK2 uses accuracy first,
discovers it "looks good" (83-84%) but minority-class F1 is only 0.64 → realises accuracy is
misleading. WK3 acts on it: switches fully to AUPRC since both accuracy AND ROC-AUC can look
deceptively good on imbalanced data. WK4 goes one step further: even AUPRC needs measuring
TWICE — once via cross-validation (can be falsely optimistic if data was resampled), once on the
REAL test set never resampled — these two numbers can diverge wildly (0.9639 vs 0.7951). See
[[auprc]].

### 2. Random Forest (WK2 baseline) vs XGBoost (WK3–WK4 main model)

Both are tree ensembles but differ in mechanism: Random Forest = **bagging** (many independent
trees, trained in parallel, majority vote); XGBoost = **boosting** (sequential trees, each
correcting the previous one's errors). On the same data (WK3/WK4's 9-model comparison), XGBoost
(0.7976 AUPRC) clearly beats Random Forest (0.7141 AUPRC) — empirical evidence that boosting
typically outperforms bagging on hard/imbalanced problems, and the reason XGBoost is chosen as
the central model for both SHAP AND FairLearn in WK4. See [[xgboost]], [[random-forest]].

### 3. Handling class imbalance — 4 tiers of cost, and who really wins (full conclusion in WK4)

(1) **`class_weight='balanced'`/`scale_pos_weight`** — doesn't alter the data, only changes
learning weights, cheapest. (2) **SMOTE/ADASYN/Undersampling** — alters the data via instant
geometric computation, medium cost. (3) **PCA Features** — not a true imbalance-handling
technique, but WK4 includes it in the same comparison grid as a data-representation variant. (4)
**CTGAN** — trains a GAN, most expensive. WK3 concludes SMOTE wins on CV (0.959); the full
54-combination grid in WK4 shows CTGAN (0.9639) edging out SMOTE (0.959) on CV — but when checked
on the REAL TEST SET, CTGAN (0.7951) actually LOSES to the model with NO treatment at all
(0.8024)! Final conclusion: on THIS specific problem, there's no solid evidence that ANY
resampling technique improves real performance over simply using `scale_pos_weight` — the
pivotal lesson of the whole course. See [[ctgan]], [[imbalanced-data-resampling]].

### 4. Data leakage — a recurring theme in multiple forms, from WK2 to WK4

WK2: leakage if `ColumnTransformer.fit()` runs on the combined/test set instead of train only —
see [[wk2-data-preparation-feature-engineering]]. WK3: a subtler leakage if SMOTE resamples
BEFORE the CV fold split — a synthetic sample in the validation fold could originate from a real
sample in the training fold — see [[wk3-modelling-imbalance-deep-learning]], "Pitfalls" section.
WK4: a BROADER form of "leakage" — the entire evaluation WORKFLOW (not just one fit step) can be
falsely optimistic without a final confirmation step on data COMPLETELY held out —
XGBoost+CTGAN winning CV but losing on the real test set is a direct consequence of the same
leakage principle at a larger scale. All 3 are variants of one principle: any "learns from data"
step (fitting a scaler, fitting a resampler, tuning hyperparameters) must only run on data
genuinely used for training, and a completely untouched holdout set must always remain until the
final evaluation step.

### 5. Fairness — the question raised earliest in the course (WK1), the tooling to address it comes latest (WK4)

WK1 already flagged meaningful demographic skew BEFORE any model existed (Male 66.85%/Female
33.15%, White 85.50%) and raised the fairness question right at the data-exploration stage — see
[[wk1-data-understanding]] section 6. Throughout WK2–WK3, this question was TEMPORARILY SET
ASIDE while focus stayed on predictive accuracy. Only WK4 comes back to address it with concrete
tooling — [[fairlearn]] measures `MetricFrame` across exactly the demographic groups already
flagged as skewed in WK1 (`is_male`, `race_white/black/...`), then mitigates via
`ExponentiatedGradient`/`ThresholdOptimizer`/manual adjustment. Lesson: raising the fairness
question EARLY (WK1) doesn't mean ADDRESSING it early — but raising it early means knowing
EXACTLY which groups to measure once mitigation tooling is ready.

### 6. WK1's EDA sows the seeds for WK2's Feature Engineering

WK1's `network-graph-categorical-eda` explores the relationship between `education` and
`occupation` via a network graph — this observation isn't "just for show": WK2's
`CustomFeatureCreator` later creates a series of binary flags from `occupation`
(`occupation_executuive`, `occupation_admin`...) and `workclass`/`race` — these groupings aren't
random, they reflect intuition already formed during Week 1's exploration. Lesson: EDA (WK1)
isn't a "nice to know" step separate from modelling — it directly SHAPES the features created
later.

## Wiki Links

[[wk1-quiz]] · [[wk2-quiz]] · [[wk3-quiz]] · [[wk4-quiz]] · [[wk1-data-understanding]] ·
[[wk2-data-preparation-feature-engineering]] · [[wk3-modelling-imbalance-deep-learning]] ·
[[wk4-model-selection-explainability-fairness-deployment]] · [[overview]]
