---
type: concept
title: "Dimensionality Reduction: PCA & t-SNE"
tags: [pca, tsne, dimensionality-reduction, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**PCA** (Principal Component Analysis) and **t-SNE** (t-distributed Stochastic Neighbor
Embedding) are both **dimensionality reduction** techniques — turning many-column data into fewer
columns — but PCA is a LINEAR transform preserving global structure, while t-SNE is a NON-LINEAR
transform preserving local structure, mainly used for visualisation.

## Intuition / Why It Matters

Data after feature engineering has 19 columns — many of which may carry REDUNDANT information
(e.g. `education-num` and the `occupation_*` flags partly correlate). Dimensionality reduction
helps: (1) faster training, less overfitting with fewer dimensions; (2) VISUALISING
high-dimensional data by eye (only 2D/3D can be plotted); (3) discovering hidden structure
(natural data groupings) before attempting clustering.

## Mechanism

**PCA**: finds axes (principal components) along the direction of GREATEST data variation,
subsequent axes orthogonal to previous ones with decreasing variation — keeping the top K
components retains the K largest chunks of variance. Component count is chosen by an
EXPLAINED-VARIANCE THRESHOLD (e.g. 95%) — the notebook needs 10/19 components to reach 95%.
**t-SNE**: for every pair of points, computes a "closeness" probability in the original
(high-dimensional) space and the target (2D) space, then optimises point positions in the target
space so those two probability distributions are AS SIMILAR AS POSSIBLE — points close together
in the original data tend to stay close after reduction, but absolute DISTANCES on the plot carry
no reliable quantitative meaning.

## Example

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA — choose the number of components by % explained variance
pca = PCA().fit(Train_X_transformed)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_n = np.argmax(cumulative_variance >= 0.95) + 1   # = 10 in the original notebook

pca_optimal = PCA(n_components=optimal_n)
Train_X_pca = pca_optimal.fit_transform(Train_X_transformed)   # (35065, 10)

# t-SNE — for visualisation only, NOT used as model input
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Train_X_tsne = tsne.fit_transform(Train_X_transformed)
plt.scatter(Train_X_tsne[:, 0], Train_X_tsne[:, 1], c=Train_y_transformed, cmap='viridis')
```

## Where It Appears

Appears in [[wk4-notebook]]/"Modelling Part IV" — PCA is used both for visualisation (loadings
heatmap, PC2 vs PC3 scatter) AND as one of the 54-combination grid's 6 strategies
(`n_components=9`, differing from the 10 components found in the 95%-variance exploration step —
a notable, not-fully-matching detail). t-SNE is used only for visualisation (tried on both raw
data and after PCA), not part of the model comparison grid.

## Related Concepts

PCA vs t-SNE: PCA is fast, has an explicit mathematical formula, GENERALISES to new data
(`.transform()`), suits use as a model preprocessing step; t-SNE is much slower, has no
closed-form mapping for new points (must rerun entirely), suits only exploratory visualisation.
Both differ from [[kmeans-dbscan-clustering]] — dimensionality reduction transforms the data's
COORDINATE AXES, while clustering assigns a GROUP LABEL to each point; they are often used IN
SEQUENCE (reduce dimensions first, then cluster/visualise).

## Wiki Links

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[kmeans-dbscan-clustering]] · [[anomaly-detection]] · [[xgboost]]
