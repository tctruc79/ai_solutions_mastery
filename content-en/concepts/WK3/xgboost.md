---
type: concept
title: "XGBoost"
tags: [algorithm, gradient-boosting, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**XGBoost** (Extreme Gradient Boosting) is a SEQUENTIAL tree-ensemble algorithm — each new tree
is trained to correct the errors (loss-function gradient) left by the previous trees, unlike
Random Forest which builds trees independently in parallel ([[random-forest]]).

## Intuition / Why It Matters

Because each tree "learns from the errors" of previous trees, boosting typically achieves higher
accuracy than bagging (Random Forest) on the same data, especially on hard/imbalanced problems —
exactly matching the course's result: XGBoost (0.7976) and Gradient Boosting (0.7962) lead the
9-model comparison, well ahead of Random Forest (0.7124) on the original data.

## Mechanism

2 usages in the course: (1) the sklearn-style API `XGBClassifier(...)` — familiar, easy to drop
into `cross_val_score`/`Pipeline`; (2) the native API `xgb.DMatrix` + `xgb.train(params, dtrain,
custom_metric=..., evals=...)` — allows passing a CUSTOM evaluation function (e.g. a
hand-written `auprc_eval`) tracked every boosting round, and the `scale_pos_weight = count(class
0) / count(class 1)` parameter so the model self-balances somewhat during training, without
needing resampling first.

## Example

```python
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

def auprc_eval(preds, dtrain):
    labels = dtrain.get_label()
    precision, recall, _ = precision_recall_curve(labels, preds)
    return 'auprc', float(auc(recall, precision))

scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
params = {'objective': 'binary:logistic', 'scale_pos_weight': scale_pos_weight,
          'eval_metric': 'logloss', 'seed': 42}

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=100,
                   evals=[(dtrain, 'train')], custom_metric=auprc_eval)

feature_importance = model.get_score(importance_type='gain')  # more meaningful than 'weight'/'cover'
```

## Where It Appears

Becomes the CENTRAL model from [[wk3-notebook]] onward: deep-dived in WK3 (combined with
[[imbalanced-data-resampling]] + [[hyperopt-tpe]]), remains the top contender in the
54-combination grid in [[wk4-notebook]] (winning with XGBoost + CTGAN Oversampling, AUPRC ≈
0.964 on CV, but ~0.80 on the real test set — an important lesson on proper evaluation), and is
the model chosen for [[shap]] explainability and [[fairlearn]] fairness checks.

## Related Concepts

See [[random-forest]] for the bagging-vs-boosting comparison (Random Forest reaches 0.7141 AUPRC
on the original data in the same 9-model comparison — clearly behind both XGBoost and Gradient
Boosting). Versus **Gradient Boosting** (plain sklearn version, also in the 9-model table) —
XGBoost adds regularisation (L1/L2), automatic missing-value handling, and faster training — the
two give close results (0.7976 vs 0.7962 AUPRC) since they share the same boosting principle.

## Wiki Links

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[random-forest]] · [[auprc]] ·
[[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[wk4-notebook]]
