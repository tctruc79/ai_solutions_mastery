---
type: concept
title: "Point-Biserial Correlation"
tags: [statistics, correlation, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**Point-biserial correlation** measures the linear relationship between a continuous variable and
a binary (2-value) variable — mathematically it is a special case of Pearson correlation when one
of the two variables is coded 0/1.

## Intuition / Why It Matters

Standard Pearson correlation assumes both variables are continuous — not valid when one variable
is a categorical label like `income` (`<=50K`/`>50K`). Point-biserial correlation handles exactly
this case without leaving the familiar Pearson framework.

## Mechanism

`scipy.stats.pointbiserialr(binary_var, continuous_var)` returns an r coefficient in [-1, 1],
interpreted exactly like ordinary Pearson r — a positive sign means the binary variable's value
`1` co-occurs with higher values of the continuous variable.

## Example

```python
from scipy.stats import pointbiserialr

all_features['income_nr'] = (all_features['income'] == '>50K').astype(int)
corr, _ = pointbiserialr(all_features['income_nr'], all_features['age'])
# corr ≈ 0.178 — the older someone is, the (weakly) higher their probability of >50K income
```

## Where It Appears

[[wk1-notebook]] section 3.2 (first application, comparing `age`, `education-num`,
`capital-gain`, `capital-loss`, `hours-per-week` against `income`) — reused as-is by
[[wk1-practice]] for the Bank Marketing dataset (`duration`, `pdays`, `previous`, `campaign`
against target `y`).

## Related Concepts

When the target has >2 classes (not binary), use **ANOVA** or **Cramér's V** (for two categorical
variables) instead of point-biserial — both are listed in the original notebook but not
implemented in detail. When you need to measure a variable's importance WITHIN a specific model
(not just standalone correlation), use Random Forest feature importance ([[random-forest]]) or
[[shap]] — these capture non-linear interactions between multiple variables that point-biserial
(1-to-1, linear only) misses.

## Wiki Links

[[wk1-notebook]] · [[wk1-data-understanding]] · [[wk1-practice]] · [[random-forest]] · [[shap]]
