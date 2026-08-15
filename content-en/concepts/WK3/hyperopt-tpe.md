---
type: concept
title: "Hyperopt & TPE (Tree-structured Parzen Estimator)"
tags: [hyperparameter-tuning, hyperopt, tpe, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**Hyperopt** is a Python hyperparameter-tuning library; the algorithm used by default in the
course is **TPE** (Tree-structured Parzen Estimator) — a Bayesian-style search method that
INTELLIGENTLY picks the next hyperparameter set based on previous trial results, instead of
sweeping an even grid (grid search) or picking purely at random (random search).

## Intuition / Why It Matters

XGBoost has many CONTINUOUS hyperparameters (`learning_rate`, `subsample`,
`colsample_bytree`...) — grid search needs to discretise each into a few levels then sweep the
PRODUCT of all combinations, with cost growing exponentially in the number of parameters. Random
search is cheaper but learns nothing from previous trials — it can waste time re-trying regions
already known to be poor. TPE balances both: as cheap as random search (no full grid sweep
needed) yet still LEARNS from history like a directed method.

## Mechanism

TPE splits existing trials into 2 groups by outcome (good / worse than a threshold), then models
the DISTRIBUTION of hyperparameters within each group — parameter regions that appear OFTEN in
the "good" group and RARELY in the "poor" group are prioritised for the next trial. In the
course: `hp.quniform('n_estimators', 50, 300, 10)` defines a discrete search space with step 10;
`hp.uniform('learning_rate', 0.01, 0.3)` defines a continuous space; the `objective(params)`
function returns `{'loss': -score, 'status': STATUS_OK}` (Hyperopt ALWAYS minimises, so AUPRC —
where higher is better — must be NEGATED); `fmin(fn=objective, space=param_space,
algo=tpe.suggest, max_evals=50, trials=Trials())` runs 50 trials, with `Trials()` storing
history so TPE learns progressively across rounds.

## Example

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

def objective(params):
    model = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
    )
    score = cross_val_score(model, X_train, y_train, cv=skf,
                             scoring='average_precision').mean()
    return {'loss': -score, 'status': STATUS_OK}   # negated since Hyperopt always minimizes

param_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
}

best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest,
                    max_evals=50, trials=Trials())
```

## Where It Appears

Used in [[wk3-notebook]]/"XGBoost for Imbalanced Data" to fine-tune XGBoost on the
SMOTE-resampled set (see [[imbalanced-data-resampling]]) — raising AUPRC from 0.959 (default
parameters) to 0.9635 after 50 trials (~37 minutes of actual runtime), the resulting model used
for comparison in [[wk4-notebook]].

## Related Concepts

**Grid Search**: guaranteed to find the best point WITHIN the defined grid, but cost grows
exponentially — infeasible with many continuous parameters. **Random Search**: cheap, easy to
implement, but doesn't leverage information from previous trials. **TPE (Hyperopt)**: balances
the two extremes above — learns from history to focus the search on promising regions, well
suited when the parameter space is large/continuous and each trial is expensive (like training
XGBoost through 10-fold CV).

## Wiki Links

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[xgboost]] ·
[[imbalanced-data-resampling]] · [[auprc]]
