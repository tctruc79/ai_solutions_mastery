---
type: concept
title: "Unsupervised Clustering: K-Means & DBSCAN"
tags: [kmeans, dbscan, clustering, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**K-Means** and **DBSCAN** are both unsupervised **clustering** algorithms — automatically
grouping SIMILAR data points together WITHOUT using the `income` label — but follow 2 entirely
different philosophies: K-Means partitions the space into K spherical regions around centroids,
DBSCAN groups by point DENSITY.

## Intuition / Why It Matters

After already having a good supervised classifier (XGBoost), another worthwhile question: does
the data naturally split into different GROUPS OF PEOPLE, without knowing the income label in
advance? Clustering helps uncover this hidden structure — e.g. one cluster might be "young, few
hours worked, low income" while another is "older, highly educated, high income" — useful for
segmentation or deeper data understanding beyond pure prediction.

## Mechanism

**K-Means**: choose the cluster count K upfront, initialise K random centroids, ITERATE 2 steps
to convergence: (1) assign each point to its NEAREST centroid, (2) update each centroid = mean of
its assigned points. K is chosen via the **elbow method** — plotting within-cluster sum of
squares (inertia) against K, picking the "elbow" point where adding more clusters no longer
meaningfully reduces inertia. **DBSCAN**: for each point, count neighbours within radius `eps` —
if it has enough (`min_samples`) neighbours, it's a **core point** and starts/extends a cluster; a
point with too few neighbours but near a core point is a **border point**; everything else is
**noise** (label `-1`) — DBSCAN determines the cluster count itself, no K needed.

## Example

```python
from sklearn.cluster import KMeans, DBSCAN

# K-Means — elbow method to choose K
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
            for k in range(1, 25)]
plt.plot(range(1, 25), inertias)   # find the "elbow" point

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Train_X_transformed)
cluster_income_rate = pd.Series(Train_y_transformed).groupby(kmeans.labels_).mean()

# DBSCAN — no K needed, self-detects noise
dbscan = DBSCAN(eps=0.5, min_samples=40)
labels = dbscan.fit_predict(Train_X_transformed)
n_noise = (labels == -1).sum()   # number of points treated as noise, belonging to no cluster
```

## Where It Appears

[[wk4-notebook]]/"Modelling Part IV": K-Means uses the elbow method then SELECTS `optimal_k = 10`
— despite a code comment reading "let's say it appears to be 3" (not matching the code that
actually runs — see [[wk4-notebook]]'s quirks section). DBSCAN uses `eps=0.5, min_samples=40`,
dropping noise points before computing the average `>50K` rate per remaining cluster. Both are
used only for exploration, not fed into the 54-combination classification-model comparison grid.

## Related Concepts

K-Means vs DBSCAN: K-Means is fast, easy to understand, but needs K known upfront and assumes
roughly spherical clusters (not suited to irregularly shaped clusters); DBSCAN is more flexible
about cluster shape and self-detects noise/outliers, but is sensitive to the `eps`/`min_samples`
choice and handles clusters of very DIFFERENT density poorly. Different from
[[hierarchical-clustering-categorical-grouping]] (WK2) — WK2 clusters the LEVELS of one
categorical variable (occupation) based on the target rate, while K-Means/DBSCAN here cluster
INDIVIDUAL DATA ROWS (people) based on the full numeric feature set. Related to
[[anomaly-detection]] — DBSCAN's "noise" concept is close to the anomaly-detection idea, though
designed for a different purpose (clustering vs flagging outlier points).

## Wiki Links

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[pca-tsne-dimensionality-reduction]] · [[anomaly-detection]] ·
[[hierarchical-clustering-categorical-grouping]]
