---
type: concept
title: "Handling Class Imbalance: SMOTE, ADASYN, Random Undersampling"
tags: [imbalanced-data, smote, adasyn, undersampling, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

3 **resampling** techniques to rebalance class ratios before training: **SMOTE** (Synthetic
Minority Oversampling Technique — generates NEW minority samples via interpolation), **ADASYN**
(Adaptive Synthetic Sampling — a SMOTE variant that generates more samples in hard-to-classify
regions), and **Random Undersampling** (randomly drops majority samples, generates nothing new).

## Intuition / Why It Matters

A model trained on imbalanced data (the `>50K` class is only 24%) tends to favour the majority
class, because minimising overall loss is easier by "defaulting" to the majority class for
ambiguous cases. Rebalancing the class ratio in the TRAINING set (not the test set) forces the
model to learn the minority class's distinguishing features instead of ignoring it.

## Mechanism

**SMOTE**: for each minority sample, find its k nearest same-class neighbours, randomly pick
one, create a new point ON THE LINE SEGMENT joining the two (linear interpolation) — repeat
until minority count matches majority count. **ADASYN**: like SMOTE but DISTRIBUTES the
generated samples unevenly — generates MORE around minority samples surrounded by many
majority-class neighbours (i.e. in the hard-to-classify border region), fewer in purely minority
regions (already easy to classify). **Random Undersampling**: randomly selects a subset of the
majority class sized to match the minority class, KEEPS the entire minority class, drops the
rest of the majority — generates no new data but may discard useful information.

## Example

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# SMOTE — each technique returns a new, balanced (X, y)
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)
X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

print(y_train.value_counts())       # before: 26510 vs 8555 (skewed)
print(pd.Series(y_smote).value_counts())  # after SMOTE: 26510 vs 26510 (balanced)
```

## Where It Appears

Central to [[wk3-notebook]]/"XGBoost for Imbalanced Data": SMOTE wins clearly (AUPRC ≈ 0.959)
over ADASYN (≈0.949) and Undersampling on this problem, and is chosen for the following
[[hyperopt-tpe]] tuning step. [[wk4-notebook]] expands this into one of 6 imbalance-handling
strategies tested across the full 9-model grid, alongside [[ctgan]] (a completely different
generation mechanism).

## Related Concepts

SMOTE vs ADASYN: same linear-interpolation mechanism, differing in HOW generated samples are
DISTRIBUTED (even vs border-focused) — ADASYN theoretically helps the model learn harder regions
better, but on this problem SMOTE yields higher AUPRC, a reminder that "more sophisticated"
doesn't always mean "better" on every dataset. Oversampling (SMOTE/ADASYN) vs Undersampling:
oversampling preserves all majority-class information but grows the dataset (slower training);
undersampling is faster but discards data. Fundamentally different from [[ctgan]] —
SMOTE/ADASYN only interpolate GEOMETRICALLY between real points, while CTGAN trains a generator
network to relearn the entire data distribution.

## Wiki Links

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[xgboost]] · [[auprc]] ·
[[hyperopt-tpe]] · [[ctgan]] · [[wk4-notebook]]
