---
type: self-test
title: "WK4 Self-Test — Model Selection, Explainability, Fairness, Deployment"
tags: [self-test, wk4, shap, fairlearn, pca]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Scope: [[wk4-model-selection-explainability-fairness-deployment]], [[wk4-notebook]]. Practice
questions authored by Claude, not an official exam.

## Q1

XGBoost + CTGAN wins the 54-combination grid with a 0.9639 cross-validation AUPRC, but only
reaches 0.7951 on the real test set — LOWER than non-resampled XGBoost (0.8024). Explain why.

**Answer**: Synthetic samples (SMOTE/CTGAN) tend to be "closer to" and "easier to predict" than
real data since they're generated from those very points — cross-validation measured on a set
mixed with these easy synthetic samples produces an artificially inflated score. The real test set
(never resampled) is what actually reflects performance on genuinely new data. See
[[wk4-model-selection-explainability-fairness-deployment]] section 1.

## Q2

The original notebook prints "Best model based on AUPRC: Extra Trees SMOTE with an AUPRC of
0.9026" but the correctly sorted table right below shows the real winner is XGBoost + CTGAN
(0.9639). Where exactly is the technical bug?

**Answer**: `max(results, key=lambda x: x[1])` iterates over the dict's KEYS (model-name strings),
not (key, value) pairs — `x[1]` is therefore the model name's 2nd character, not an AUPRC score.
Correct fix: `max(results.items(), key=lambda kv: kv[1][0])`. See [[wk4-notebook]] quirk #1.

## Q3

PCA needs 10/19 components to retain 95% of variance. What does this say about the original 19
features?

**Answer**: The 19 original features have considerable REDUNDANCY (overlapping information) — you
don't need all 19 dimensions to capture most of the information, over half the dimensions can be
"compressed away" without losing much. See [[pca-tsne-dimensionality-reduction]].

## Q4

K-Means needs K known in advance, DBSCAN doesn't. So how does DBSCAN "know" the cluster count, and
what concept does it have that K-Means lacks?

**Answer**: DBSCAN determines the cluster count itself based on point DENSITY (`eps`,
`min_samples`) — clusters form from sufficiently dense regions. It has a NOISE concept (label -1)
for points belonging to no cluster, while K-Means MUST assign every point to some cluster. See
[[kmeans-dbscan-clustering]].

## Q5

How do SHAP "global" (`summary_plot`) and SHAP "local" (`waterfall`) answer different questions?
Give an example situation needing local instead of global.

**Answer**: Global answers "which feature matters most OVERALL across the whole dataset?"; local
answers "why did the model predict THIS SPECIFIC row the way it did?". Local is needed when a
specific decision must be justified — e.g. explaining to one customer why their loan application
was rejected, global can't answer that individual-level question. See [[shap]].

## Q6

FairLearn offers 3 intervention approaches: in-processing, post-processing, and manual
adjustment. Where do they differ in the pipeline, and which one does NOT require retraining the
model?

**Answer**: In-processing (`ExponentiatedGradient`) bakes the fairness constraint into training
itself — requires retraining. Post-processing (`ThresholdOptimizer`) only adjusts the
classification THRESHOLD after the model already exists — does NOT require retraining. Manual
adjustment also doesn't require retraining but lacks the mathematical guarantees of the two
approaches above. See [[fairlearn]].

## Q7

The final model `joblib.dump`ed for deployment was Hyperopt-tuned on CTGAN data (0.9653 CV AUPRC)
— did the notebook re-confirm THIS tuned model's AUPRC on the real test set? Why is this a
notable gap?

**Answer**: No — the "Testing on Test Set" section only checks 2 UNTUNED models (original: 0.8024,
CTGAN: 0.7951); the tuned model is never re-measured on the real test set in the notebook. This is
a notable gap because the section right above it just proved high CV doesn't guarantee high test
performance — skipping this final confirmation means the DEPLOYED model's true performance
remains unknown with certainty. See [[wk4-notebook]] quirk #2.

## Wiki Links

[[wk4-model-selection-explainability-fairness-deployment]] · [[wk4-notebook]] · [[wk3-quiz]] ·
[[cumulative-quiz]]
