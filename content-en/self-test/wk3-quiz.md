---
type: self-test
title: "WK3 Self-Test — Modelling II: 9 Models, XGBoost, Imbalance, Deep Learning"
tags: [self-test, wk3, xgboost, imbalanced-data]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Scope: [[wk3-modelling-imbalance-deep-learning]], [[wk3-notebook]]. Practice questions authored
by Claude, not an official exam.

## Q1

Why is AUPRC chosen as the official metric from this week on, instead of ROC-AUC (which is also
considered more robust to imbalance than accuracy)?

**Answer**: ROC-AUC factors in the True Negative Rate — with the negative class at 76% of the
data, TNR stays high almost regardless of how poor the model is on the minority class, so
ROC-AUC can still look deceptively good. AUPRC only uses Precision and Recall, both computed ON
the positive (minority) class, so it more honestly reflects real performance on the class that
matters. See [[auprc]].

## Q2

`scale_pos_weight` in XGBoost and applying SMOTE before training both "handle imbalance" — where
do they actually differ in the workflow?

**Answer**: `scale_pos_weight` adjusts the LEARNING WEIGHT during training itself — the input
data doesn't change size, the model just "pays more attention" to the minority class when
computing loss. SMOTE changes the DATA ITSELF before the model ever sees it — generating extra
minority samples so the class ratio is balanced from the input onward. Both can be used
independently or combined. See [[xgboost]] and [[imbalanced-data-resampling]].

## Q3

SMOTE, ADASYN, and Random Undersampling all "balance classes" but through different mechanisms.
Briefly describe each mechanism.

**Answer**: SMOTE — linearly interpolates between a minority sample and its nearest same-class
neighbour to create a new sample. ADASYN — like SMOTE but generates more samples in border
regions (minority samples surrounded by many majority neighbours). Random Undersampling —
randomly drops majority samples to match the minority count, generates nothing new. See
[[imbalanced-data-resampling]].

## Q4

The first Deep Learning model in the notebook produces `loss: nan` throughout all 20 epochs.
What's the exact technical cause, and the fix?

**Answer**: The final layer uses `activation='tanh'` (range [-1, 1]) but the loss is
`categorical_crossentropy` (requires probabilities in [0, 1]) — this wrong pairing causes `nan`
from the very first round. Fix by changing the final activation to `sigmoid` (binary, 1 output)
or `softmax` (multi-class) to match cross-entropy loss. See
[[wk3-modelling-imbalance-deep-learning]] section 6.

## Q5

The original notebook runs `smote.fit_resample()` on the ENTIRE training set ONCE, then
`cross_val_score` over 10 folds on that already-resampled data. Why does this carry a risk, and
what's the more correct approach?

**Answer**: A SMOTE synthetic sample in the validation fold may have been interpolated from a
real sample sitting in the train fold (since resampling happens BEFORE the fold split) — so the
validation fold is no longer fully "unseen", potentially inflating the AUPRC score. The more
correct approach: put SMOTE inside an `imblearn.pipeline.Pipeline`, so resampling only uses that
fold's own training portion. See [[wk3-modelling-imbalance-deep-learning]], "Pitfalls" section.

## Q6

XGBoost uses `importance_type='gain'` to compute feature importance instead of `'weight'`.
What's the difference, and why is `gain` more appropriate?

**Answer**: `weight` counts the NUMBER OF TIMES a feature is used to split across all trees —
only reflects FREQUENCY. `gain` measures the ACTUAL LOSS REDUCTION that feature contributes each
time it's used — more accurately reflecting IMPORTANCE, since a feature might be used rarely but
contribute a large loss reduction each time. See [[xgboost]].

## Wiki Links

[[wk3-modelling-imbalance-deep-learning]] · [[wk3-notebook]] · [[wk2-quiz]] · [[cumulative-quiz]]
