---
type: lecture
title: "WK3 — Modelling II: Comparing 9 Models, XGBoost, Handling Imbalance & Deep Learning"
tags: [modelling, xgboost, imbalanced-data, hyperopt, deep-learning, ctgan, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE - ADASync - CTGANs.ipynb"
---

## Position in the Course

Follows directly from [[wk2-data-preparation-feature-engineering]] — last week's
feature-engineering pipeline is REUSED UNCHANGED, and the closing lesson of Week 2 ("83-84%
accuracy is an illusion under class imbalance") becomes this week's STARTING POINT: everything
here is designed to answer "so what's the right metric, and how do we actually fix the
imbalance?". Week 4 expands this into a full grid of 9 models × 6 imbalance-handling strategies,
reusing every technique learned here.

## Learning Objectives

After this lesson you can: (1) explain why AUPRC suits an imbalanced problem better than
accuracy/ROC-AUC, and know its random baseline; (2) objectively compare multiple models via
cross-validation on the same metric; (3) use XGBoost through both APIs (sklearn-style and
native) with a custom eval metric; (4) distinguish the mechanisms of SMOTE/ADASYN/Random
Undersampling and know when each is appropriate; (5) use Hyperopt (TPE) for smarter
hyperparameter search than grid search; (6) recognise a common neural-network architecture bug
(final activation mismatched with the loss function) when you see `loss: nan`; (7) understand
how CTGAN's synthetic-data mechanism differs from SMOTE's.

## Core Theory

### 1. Why switch entirely to AUPRC as the official metric

Week 2 stopped at the observation "accuracy is misleading"; Week 3 acts on it: EVERY model
comparison from here uses `make_scorer(average_precision_score, needs_proba=True)` — see
[[auprc]]. The deeper reason: even ROC-AUC (usually considered more robust than accuracy) can
still look deceptively good on imbalanced data because it factors in the True Negative Rate, and
with the negative class at 76%, TNR stays high almost by default. AUPRC focuses entirely on the
positive class, so it's more honest.

### 2. Comparing 9 models — an objective, repeatable process

Instead of training each model separately and eyeballing the comparison, the notebook packages
all 9 models (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
XGBoost, Extra Trees, AdaBoost, BaggingClassifier) into one dict and runs the SAME
`cross_val_score` loop with the SAME `StratifiedKFold(n_splits=10)` and the SAME AUPRC scorer —
guaranteeing a fair comparison (same data, same fold split, same metric). Result: XGBoost
(0.7976) and Gradient Boosting (0.7962) lead; Decision Tree (0.5069) is weakest, close to Extra
Trees (0.6079).

### 3. XGBoost — two ways to use it, and why the native API when deeper control is needed

[[xgboost]] has 2 APIs: `XGBClassifier` (sklearn-style, convenient for
`cross_val_score`/`Pipeline`, used in the 9-model comparison) and the native `xgb.DMatrix` +
`xgb.train(params, dtrain, custom_metric=..., evals=...)` (used in the XGBoost deep-dive) — the
native API allows passing `custom_metric=auprc_eval` (a hand-written function, tracked EVERY
boosting round via `evals_result`) and `scale_pos_weight = count(class 0) / count(class 1)` so
XGBoost self-adjusts its learning weight during training — a way to handle imbalance WITHOUT
resampling first. Feature importance uses `importance_type='gain'` (actual contribution to loss
reduction) rather than `'weight'` (how many times a tree used that feature) — `gain` more
accurately reflects IMPORTANCE rather than FREQUENCY.

### 4. Three resampling techniques — different mechanisms, not just "balancing the count"

[[imbalanced-data-resampling]] covers 3 techniques with entirely different mechanisms despite
sharing the goal of balancing classes: **SMOTE** (generates new minority samples via linear
interpolation between a point and its nearest same-class neighbour), **ADASYN** (like SMOTE but
generates MORE samples in hard-to-classify border regions), and **Random Undersampling**
(randomly drops majority samples, generates nothing new). Empirical result: XGBoost + SMOTE
reaches AUPRC ≈ 0.959, well ahead of ADASYN (≈0.949) and the original data — SMOTE is chosen
(`best_X_train, best_y_train = Train_X_smote, Train_y_smote`) for the next tuning step.

### 5. Hyperopt (TPE) — intelligent search instead of a grid sweep

[[hyperopt-tpe]] uses the Tree-structured Parzen Estimator algorithm to pick the next
hyperparameter set BASED ON previous trial results, instead of sweeping every combination evenly
(grid search — cost grows exponentially with more continuous parameters) or picking purely at
random (random search — learns nothing from history). `fmin(fn=objective, space=param_space,
algo=tpe.suggest, max_evals=50)` raises AUPRC from 0.959 (SMOTE, default params) to 0.9635
(SMOTE + tuned) after 50 trials — taking ~37 minutes in the original notebook, a reminder that
tuning has real computational cost.

### 6. Deep Learning — the final layer's activation MUST match the loss function

The first Keras model uses `Dense(2, activation='tanh')` (range [-1, 1]) as its final layer but
compiles with `loss='categorical_crossentropy'` (which requires probabilities in [0, 1]) — this
WRONG pairing makes `loss: nan` appear from epoch 1 and never recover across all 20 epochs. This
is a genuine technical bug in the original notebook, not intentional illustration — the lesson:
for binary/multi-class classification with cross-entropy, the final layer must use `sigmoid`
(binary, 1 output) or `softmax` (multi-class, output = number of classes), NEVER `tanh`. The
notebook doesn't fix this cell but switches to a different architecture (`sigmoid` +
`binary_crossentropy`, 1 output) afterward.

### 7. CTGAN — generating synthetic data via adversarial networks, a different mechanism than SMOTE

[[ctgan]] (Conditional Tabular GAN) doesn't linearly interpolate between real points like
SMOTE — it trains a pair of adversarial neural networks (a generator producing fake samples, a
discriminator trying to tell real from fake) to RELEARN the data distribution, then samples new
points from that learned distribution — in principle this can produce more diverse samples than
linear interpolation, but costs far more training time (even at just `CTGAN(epochs=5)` — very
few epochs — it's still a neural network requiring training, unlike SMOTE's instant geometric
computation). One illustrative fold: CTGAN Oversampling (AUPRC ≈ 0.728) doesn't clearly beat the
Base Model (≈ 0.732) in this run — the full 5-fold output wasn't preserved in the original
notebook, so CTGAN's win/loss at this step CANNOT be concluded from a single fold alone.

## Worked Example

```python
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb_base_params = {'random_state': 42, 'eval_metric': 'logloss',
                    'scale_pos_weight': scale_pos_weight}

# Compare 3 resampling techniques by AUPRC
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)
X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

for name, (X_r, y_r) in {'SMOTE': (X_smote, y_smote), 'ADASYN': (X_adasyn, y_adasyn),
                          'Undersample': (X_under, y_under)}.items():
    score = cross_val_score(xgb.XGBClassifier(**xgb_base_params), X_r, y_r,
                             cv=skf, scoring='average_precision').mean()
    print(f'{name}: {score:.4f}')   # SMOTE wins clearly in the original notebook (~0.959)

# Hyperopt TPE tuning on the winning strategy (SMOTE)
def objective(params):
    model = xgb.XGBClassifier(**xgb_base_params,
                               n_estimators=int(params['n_estimators']),
                               max_depth=int(params['max_depth']),
                               learning_rate=params['learning_rate'])
    score = cross_val_score(model, X_smote, y_smote, cv=skf,
                             scoring='average_precision').mean()
    return {'loss': -score, 'status': STATUS_OK}   # Hyperopt ALWAYS minimizes -> negate AUPRC

space = {'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
         'max_depth': hp.quniform('max_depth', 3, 15, 1),
         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)}
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials())
```

## Pitfalls & Practical Notes

- **Resampling BEFORE cross-validating** (as the original notebook does: `Train_X_smote =
  smote.fit_resample(...)` once on the whole train set, then `cross_val_score(cv=skf)` on the
  already-resampled data) — carries a SUBTLE leakage risk: a SMOTE synthetic sample in the
  validation fold may have been interpolated from a real sample sitting in the train fold, so
  the validation fold is no longer fully "unseen". The more correct approach: put SMOTE INSIDE
  each fold (e.g. `imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', model)])`, then
  `cross_val_score` on that Pipeline) so resampling only uses information from that fold's
  training portion.
- **`Dense(N, activation='tanh')` + `categorical_crossentropy`** — whenever you see `loss: nan`
  from the very first epoch, check the match between the final activation and the loss function
  FIRST, before suspecting the learning rate or the data.
- **Comparing models across different metrics between steps** (accuracy in WK2, AUPRC in WK3) —
  only valid when you clearly understand WHY the metric changed; don't directly compare 83%
  accuracy (WK2) with 0.80 AUPRC (WK3) as if they were on the same scale.

## Related Concepts

[[auprc]] · [[xgboost]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[random-forest]] · [[logistic-regression]]

## Quick Self-Check

1. Why is AUPRC better suited than even ROC-AUC (not just accuracy) for this week's imbalanced
   problem?
2. `scale_pos_weight` in XGBoost and resampling (SMOTE/ADASYN) both "handle imbalance" — where
   in the training process do they actually differ?
3. The first Deep Learning model produces `loss: nan` — what's the exact technical cause?

Full question bank: [[wk3-quiz]].

## Wiki Links

[[wk3-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[wk2-notebook]] ·
[[auprc]] · [[xgboost]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[random-forest]] · [[logistic-regression]] · [[overview]]
