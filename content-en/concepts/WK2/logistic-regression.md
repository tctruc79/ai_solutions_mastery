---
type: concept
title: "Logistic Regression"
tags: [algorithm, linear-model, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

A linear model for classification — estimates the probability of the positive class via a
sigmoid function applied to a linear combination of input features:
`P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))`.

## Intuition / Why It Matters

Simple, fast to train, and every coefficient `β` is directly interpretable: sign shows whether
the feature increases/decreases the positive-class probability, magnitude shows the strength — an
advantage Random Forest lacks ([[random-forest]]). Because it's linear, it makes a good baseline
to judge whether a more complex model is actually worth the extra effort.

## Mechanism

`LogisticRegression(random_state=42, max_iter=1000)` — `max_iter` is raised because the optimiser
(default `lbfgs`) needs enough iterations to converge on scaled data (`StandardScaler`, see
[[sklearn-column-transformer-pipeline]]) — input scaling matters MUCH more for Logistic
Regression than for Random Forest, since the optimiser is sensitive to each variable's scale.

## Example

```python
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(Train_X_transformed, Train_y_transformed)
# accuracy 0.84, F1 on the >50K class = 0.64 — nearly on par with Random Forest in week 2
```

## Where It Appears

[[wk2-notebook]] (baseline, paired with [[random-forest]]) → WK3 (one of 9 compared models,
adding `class_weight='balanced'` to soften class-imbalance effects).

## Related Concepts

See [[random-forest]] for a direct comparison (non-linear vs linear, coefficient interpretation
vs feature importance).

## Wiki Links

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[random-forest]] ·
[[sklearn-column-transformer-pipeline]]
