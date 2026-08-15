---
type: concept
title: "AUPRC (Area Under the Precision-Recall Curve)"
tags: [metric, evaluation, imbalanced-data, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**AUPRC** (also called AUC-PR, or `average_precision_score` in sklearn) is the area under the
Precision-Recall curve — measures a model's ability to hold both precision (few false alarms)
AND recall (few misses) simultaneously, aggregated across every possible classification
threshold.

## Intuition / Why It Matters

On imbalanced data (the `>50K` class is only 23.9%), **accuracy is misleading** (see
[[wk2-data-preparation-feature-engineering]]) and even **ROC-AUC** can look deceptively good
because the ROC curve is dominated by the True Negative Rate — with the negative class an
overwhelming majority, TNR stays high almost regardless of how bad the model is on the minority
class. AUPRC focuses ENTIRELY on the positive (minority) class, so it more honestly reflects
real performance on an imbalanced problem.

## Mechanism

For every possible classification threshold, compute the corresponding (Precision, Recall) →
connect those points into a curve → AUPRC = area under that curve. AUPRC's random baseline
(unlike ROC-AUC's 0.5) EQUALS the positive-class rate in the data — e.g. with `>50K` = 23.9%, a
purely random model only reaches AUPRC ≈ 0.239, so AUPRC must be compared against this floor,
not against 0.5 or 1.0.

## Example

```python
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

auprc_scorer = make_scorer(average_precision_score, needs_proba=True)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring=auprc_scorer)
print(scores.mean())   # compare against the random baseline ≈ positive-class rate (0.239), NOT 0.5
```

## Where It Appears

Becomes the OFFICIAL metric from [[wk3-notebook]] onward (comparing 9 models, tuning XGBoost
with [[hyperopt-tpe]], the 54-combination grid in [[wk4-notebook]]). A key lesson in
[[wk4-notebook]]/Part IV: AUPRC measured on RESAMPLED data (SMOTE/CTGAN) is falsely optimistic —
always re-check on the REAL, unresampled test set.

## Related Concepts

Different from **ROC-AUC** — ROC-AUC uses both TPR and FPR (diluted by the overwhelming
negative class), while AUPRC uses Precision and Recall (both computed ON the positive class, not
diluted by the huge negative-class count) → AUPRC is far more sensitive to real minority-class
performance. Different from **F1-score** — F1 is a single point at one fixed threshold (usually
0.5), while AUPRC aggregates across EVERY threshold, independent of any specific threshold
choice.

## Wiki Links

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] ·
[[wk2-data-preparation-feature-engineering]] · [[xgboost]] · [[imbalanced-data-resampling]]
