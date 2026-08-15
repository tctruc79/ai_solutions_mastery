---
type: concept
title: "Hierarchical Clustering for Categorical Grouping"
tags: [feature-engineering, clustering, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

Uses hierarchical clustering (`scipy.cluster.hierarchy.linkage`, typically with `method='ward'`)
to group the levels of ONE high-cardinality categorical variable together, based on a continuous
metric measuring each level's behaviour relative to the target (usually the positive-class rate).

## Intuition / Why It Matters

A high-cardinality categorical variable (`occupation` has 15 levels), if fully one-hot encoded,
produces 15 sparse columns, several of which behave nearly identically relative to the target
(e.g. two occupations with near-equal high-income rates don't need two separate columns).
Clustering based on actual behaviour reduces dimensionality while keeping the most important
signal, and yields MEANINGFUL groups (nameable, interpretable) unlike abstract components such as
PCA.

## Mechanism

1. For each level of the categorical variable, compute one summary metric of its behaviour (e.g.
   rate of `income == '>50K'`).
2. `linkage(metric, method='ward')` builds a cluster tree based on distances between those metric
   values.
3. Plot a dendrogram, VISUALLY choose a sensible number of groups (no automatic threshold — a
   subjective-but-informed decision).
4. Manually label groups based on which levels sit close together on the dendrogram (e.g. the
   `occupation_cleaner` group bundles `Priv-house-serv`, `Handlers-cleaners`, `Other-service`).

## Example

```python
from scipy.cluster.hierarchy import linkage, dendrogram

occ_stats = (occupation_data.groupby('occupation')['income']
             .apply(lambda x: (x == '>50K').mean())
             .reset_index(name='high_income_ratio'))

linked = linkage(occ_stats[['high_income_ratio']], method='ward')
dendrogram(linked, orientation='right', labels=occ_stats['occupation'].values)
# -> inspect the dendrogram, manually merge into 7 groups: cleaner, repair, admin, handwork,
#    executive, support, army
```

## Where It Appears

[[wk2-notebook]] — applied to `occupation` (7 groups) and to the `relationship`+`marital-status`
combination. The resulting groups become fixed input columns for `CustomFeatureCreator`
throughout WK2–WK4, see [[sklearn-column-transformer-pipeline]].

## Related Concepts

Different from the **network graph** ([[network-graph-categorical-eda]]) which only visualises
co-occurrence frequency between two variables — hierarchical clustering here GROUPS the levels of
one variable based on behaviour toward the TARGET, directly feeding feature creation, not just
for viewing. Different from PCA ([[pca]], WK4) — PCA creates abstract components (linear
combinations of ALL numeric variables), while this technique only groups the LEVELS of a SINGLE
categorical variable, preserving interpretability.

## Wiki Links

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] ·
[[sklearn-column-transformer-pipeline]] · [[network-graph-categorical-eda]]
