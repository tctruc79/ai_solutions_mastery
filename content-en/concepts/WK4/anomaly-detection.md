---
type: concept
title: "Anomaly Detection: Isolation Forest, LOF, Autoencoder"
tags: [anomaly-detection, isolation-forest, lof, autoencoder, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**Anomaly detection** is the task of finding data points that are UNUSUAL relative to the bulk of
the data, trained WITHOUT needing pre-existing anomaly labels. The notebook tries 3 techniques:
**Isolation Forest**, **Local Outlier Factor (LOF)**, and a hand-written Keras **Autoencoder**.

## Intuition / Why It Matters

Supervised classification (XGBoost...) needs enough examples of BOTH classes to learn a decision
boundary. But many real problems (card fraud, equipment failure, network intrusion) have an
"abnormal" class that's EXTREMELY RARE or CONSTANTLY CHANGING (novel forms never seen before) —
not enough examples to train ordinary classification. Anomaly detection learns "what NORMAL looks
like" from the majority class, then flags anything sufficiently different — without needing to
know the specific shape of the anomaly in advance.

## Mechanism

**Isolation Forest**: builds many RANDOM decision trees, each splitting data by randomly picking
a feature + cut threshold — an ANOMALOUS point gets "isolated" (split off alone) AFTER FEWER cuts
than a normal point (since it's clearly different), so the average depth needed to isolate a
point becomes its anomaly score. **LOF**: compares a point's local neighbour DENSITY against the
density of its neighbours' own neighbours — a point sitting in a region much SPARSER than its
surroundings is anomalous. **Autoencoder**: a neural network learns to COMPRESS then RECONSTRUCT
its own input (encoder→bottleneck→decoder) — trained only on normal data, so it reconstructs
normal data WELL but reconstructs abnormal data POORLY (never seen that shape before) — a large
reconstruction error (MSE) = a suspicious point. All 3 are trained ONLY on the majority class
(`X_train[y_train == 0]`).

## Example

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Train ONLY on the majority class (income <=50K)
X_majority = X_train[y_train == 0]

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_majority)
iso_scores = -iso.decision_function(X_val)          # higher score = more anomalous

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True).fit(X_majority)
lof_scores = -lof.score_samples(X_val)

# Autoencoder — large reconstruction error = anomalous
autoencoder.fit(X_majority, X_majority, epochs=50, batch_size=256)
mse = np.mean(np.square(X_val - autoencoder.predict(X_val)), axis=1)
```

## Where It Appears

[[wk4-notebook]]/"Unsupervised Anomaly Detection": evaluated with a hand-written "AUPRC-like"
function (`calculate_auprc`) after converting anomaly scores to binary labels via a 5th-percentile
threshold — Isolation Forest (0.6161) edges out LOF (0.6082) and the Autoencoder (0.5753). Note:
because `precision_recall_curve` is fed BINARY (already-thresholded) labels instead of the
original CONTINUOUS scores, the Precision-Recall curve effectively collapses to just a few
discrete points — this "AUPRC" figure is less reliable than [[auprc]] computed properly (using
continuous scores via `predict_proba`).

## Related Concepts

Different from supervised classification (XGBoost, [[imbalanced-data-resampling]]) — anomaly
detection doesn't need 2 balanced classes, doesn't even need to know in advance what form the
"abnormal" class takes; it's an ALTERNATIVE direction when resampling isn't feasible (the minority
class too rare to generate trustworthy synthetic samples). Related to [[kmeans-dbscan-clustering]]
— DBSCAN's "noise" points (label -1) are conceptually close to anomaly detection, though DBSCAN is
designed for CLUSTERING, not optimised specifically for anomaly detection.

## Wiki Links

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[kmeans-dbscan-clustering]] · [[auprc]] · [[imbalanced-data-resampling]]
