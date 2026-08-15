---
type: concept
title: "SHAP (SHapley Additive exPlanations)"
tags: [shap, explainability, interpretability, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**SHAP** (SHapley Additive exPlanations) is a model-explanation method based on **Shapley value**
game theory — computing the FAIR contribution of EACH feature to EACH specific prediction of any
model (including "black-box" models like XGBoost).

## Intuition / Why It Matters

`feature_importances_` (Random Forest) or `gain` (XGBoost, see [[xgboost]]) only tell you which
features matter OVERALL, averaged across the whole dataset — they don't answer "why did the model
predict THIS SPECIFIC person has income >50K?". When a SPECIFIC decision needs to be justified
(e.g. a loan rejection explained to a customer) or a model needs DEBUGGING (catching it relying
on an unreasonable feature for a specific group of cases), you need to drill down to the level of
INDIVIDUAL PREDICTIONS — this is what SHAP does that ordinary feature importance cannot.

## Mechanism

For one specific prediction, SHAP treats each feature as a "player" in a cooperative game, and
computes that feature's contribution = a weighted average of how much the prediction changes when
adding that feature across EVERY possible subset of the remaining features (Shapley value from
game theory) — guaranteeing the **additive** property: the sum of EVERY feature's SHAP value + a
baseline value (the average prediction across all data) = EXACTLY EQUALS the actual prediction for
that point. `shap.TreeExplainer` computes this quickly and exactly for tree-based models (XGBoost,
Random Forest) via a specialised algorithm, no approximation needed unlike other model types.

## Example

```python
import shap

explainer = shap.TreeExplainer(final_xgb_model)

# Global — which features matter most across the WHOLE dataset
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)          # beeswarm
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names)

# Local — why the model predicted THIS SPECIFIC row the way it did
single_shap = explainer(X_sample.iloc[[3]])
shap.plots.waterfall(single_shap[0], max_display=19)
```

## Where It Appears

Used in [[wk4-notebook]]/"AI Explainability" on the final XGBoost model (trained on
SMOTE-resampled data) — the "Global Explainability" section (`summary_plot`, sampling 200–500
rows since computing SHAP for the whole dataset is very slow — the notebook notes "up to 30
minutes") and "Local Explainability" (`waterfall` per sample row). The SHAP explainer is
`joblib.dump`ed alongside the model for reuse in the deployed Streamlit app.

## Related Concepts

Different from `feature_importances_`/`gain` ([[xgboost]], [[random-forest]]) — those are a
SINGLE number per feature, computed across the whole model; SHAP gives a SEPARATE value for EVERY
(feature, prediction) pair, so it can be aggregated into both a global view (average |SHAP value|
across all data ≈ roughly equivalent to feature importance) AND a local one. Related to
algorithmic fairness ([[fairlearn]]) — SHAP can be used to check whether the model relies too
heavily on a sensitive feature (`is_male`, `race_*`) for a SPECIFIC group, complementing
FairLearn's aggregate fairness metrics.

## Wiki Links

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] · [[xgboost]] ·
[[random-forest]] · [[fairlearn]]
