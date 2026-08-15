---
type: concept
title: "Random Forest"
tags: [algorithm, ensemble, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**Random Forest** is an ensemble of many independent decision trees, each trained on a bootstrap
sample of the data and a random subset of features at every split; the final result is a majority
vote (classification) or average (regression) across all trees.

## Intuition / Why It Matters

A single decision tree overfits easily (fits training data too closely). Random Forest reduces
this risk by averaging/voting across MANY different trees (each trained on slightly different
data/features) — individual trees' errors tend to cancel out, while genuine signal gets
reinforced.

Chosen as the course's baseline model because: it handles non-linear relationships well, needs
little input scaling, gives `feature_importances_` for free interpretability, and performs well
"out-of-box" without much tuning.

## Mechanism

`RandomForestClassifier(random_state=42)` defaults to `n_estimators=100` trees, each trained
independently (parallelisable). `.feature_importances_` after fitting returns each feature's
average contribution to impurity reduction across all trees.

## Example

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(Train_X_transformed, Train_y_transformed)

importances = rf_model.feature_importances_       # 1 value / feature, sums to 1.0
# most important feature in week 2: 'binary__is_married' (~0.29)
```

## Where It Appears

Introduced as the baseline in [[wk2-notebook]] (versus [[logistic-regression]]). WK3 adds Random
Forest to the 9-model comparison table (typically loses to XGBoost/Gradient Boosting on AUPRC
with imbalanced data — see [[wk3-notebook]]). WK4 reruns Random Forest in the 54-combination
model×imbalance-technique grid.

## Related Concepts

Versus [[logistic-regression]]: Random Forest captures non-linear relationships/feature
interactions better, but is harder to interpret directly (no "beta coefficient" like logistic
regression) — the two give near-identical accuracy/F1 in WK2 (0.83 vs 0.84), a clearer gap
emerges once measured by AUPRC in WK3. Versus **XGBoost** ([[xgboost]], WK3) — both are tree
ensembles, but Random Forest builds trees INDEPENDENTLY in parallel (bagging) while XGBoost
builds trees SEQUENTIALLY, each correcting the previous one's errors (boosting) — boosting
typically wins on AUPRC for this course's imbalanced problem.

## Wiki Links

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[logistic-regression]] ·
[[wk3-notebook]] · [[point-biserial-correlation]]
