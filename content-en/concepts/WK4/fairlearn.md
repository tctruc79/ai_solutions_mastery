---
type: concept
title: "FairLearn — Measuring & Mitigating Algorithmic Bias"
tags: [fairlearn, fairness, bias-mitigation, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**FairLearn** is a Python library for measuring and mitigating **algorithmic bias** — checking
whether a model treats sensitive groups (`sensitive features`, e.g. gender, race, marital status)
MEANINGFULLY DIFFERENTLY, and providing intervention tools if so.

## Intuition / Why It Matters

Since [[wk1-data-understanding]], the data has already shown meaningful demographic skew (Male
66.85%, White 85.50%). A model achieving high AUPRC ON THE WHOLE DATASET can still hide a LARGE
performance gap between groups — e.g. predicting accurately for men but much worse for women. If
this model drives real decisions (loans, hiring), that gap could VIOLATE THE LAW or cause real
harm to one group — FairLearn helps DETECT this gap BEFORE deployment, and provides ways to
MITIGATE it if needed.

## Mechanism

**Measurement**: `MetricFrame(metrics=..., y_true=..., y_pred=..., sensitive_features=...)`
computes any metric (accuracy, precision, AUPRC, TPR...) SEPARATELY per value of
`sensitive_features`, exposing `.by_group` for direct comparison. **Mitigation — 3 different
intervention points in the pipeline**: (1) **In-processing** —
`ExponentiatedGradient(estimator=..., constraints=...)` RETRAINS the model with a fairness
constraint built directly into the loss; constraints include `DemographicParity` (equal
positive-prediction rate across groups — ignores true labels), `EqualizedOdds` (both TPR AND FPR
equal across groups — considers true labels), `TruePositiveRateParity` (only TPR equal). (2)
**Post-processing** — `ThresholdOptimizer(estimator=..., constraints="demographic_parity")` does
NOT retrain, just finds a SEPARATE classification threshold per group that satisfies the fairness
constraint. (3) **Manual** — choosing different per-group thresholds by hand based on observed
initial TPR/TNR (e.g. the notebook picks 0.3 for the not-married group, 0.7 for the married group)
— simplest but prone to subjectivity, with none of the mathematical guarantees of the two
approaches above.

## Example

```python
from fairlearn.metrics import MetricFrame, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

# 1. Measure first — detect any gap
mf = MetricFrame(metrics={'tpr': true_positive_rate, 'accuracy': accuracy_score},
                  y_true=y_true, y_pred=model.predict(X),
                  sensitive_features=X['binary__is_married'])
print(mf.by_group)     # TPR/accuracy per group — direct comparison

# 2. Mitigate — in-processing (retrain with a constraint)
mitigator = ExponentiatedGradient(estimator=xgb.XGBClassifier(), constraints=DemographicParity())
mitigator.fit(X, y_true, sensitive_features=X['binary__is_married'])

# 3. Mitigate — post-processing (no retraining)
optimizer = ThresholdOptimizer(estimator=model, constraints="demographic_parity",
                                predict_method='predict_proba')
optimizer.fit(X, y_true, sensitive_features=X['binary__is_married'])
```

## Where It Appears

Used in [[wk4-notebook]]/"AI Fairness" — measured across 6 sensitive features (`is_male`,
`is_married`, `race_white/black/asian-pac-islander`, `age`), then all 3 mitigation approaches
tried on the `is_married` feature. Directly connects to the demographic skew already flagged in
[[wk1-data-understanding]] — this is where the fairness question raised "early" in Week 1 FINALLY
gets addressed with concrete tooling.

## Related Concepts

The 3 main constraints have no "single correct answer" — `DemographicParity` suits contexts where
the goal is equal OPPORTUNITY regardless of true label (e.g. hiring quotas);
`EqualizedOdds`/`TruePositiveRateParity` suit contexts where the goal is equal ACCURACY across
groups (not "penalising" a group with a genuinely different true positive rate) — which constraint
to pick depends on the specific legal/ethical context, not a purely technical decision. Related to
[[shap]] — SHAP helps INVESTIGATE the cause of bias (which feature contributes abnormally for one
group), while FairLearn MEASURES the end result and MITIGATES — two complementary tools in a
responsible-AI process.

## Wiki Links

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[wk1-data-understanding]] · [[shap]] · [[xgboost]]
