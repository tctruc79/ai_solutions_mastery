---
type: concept
title: "Network Graph for Categorical EDA"
tags: [eda, networkx, visualization, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

Uses a directed graph (`networkx.DiGraph`) to visualise the relationship between two categorical
variables — each distinct value of a variable is a node, and the co-occurrence frequency of a
value pair becomes the weight of the edge connecting them.

## Intuition / Why It Matters

For categorical variables with many levels (e.g. `education` has 16 levels, `occupation` has 15),
a 16×15 crosstab is hard to read at a glance. A network graph compresses that into a picture —
edge thickness immediately shows which value pairs are most common, without scanning individual
table cells.

## Mechanism

1. `groupby([col_a, col_b]).size()` → count each pair's frequency.
2. Keep only the top-N most frequent pairs (avoids a cluttered graph from too many edges).
3. `G.add_edge(a, b, weight=count)` for each pair, `nx.spring_layout()` to auto-arrange nodes,
   `nx.draw()` + `nx.draw_networkx_edge_labels()` to render.

## Example

```python
import networkx as nx
import matplotlib.pyplot as plt

top10 = (input_features.loc[:, ['education', 'occupation']].dropna()
         .groupby(['education', 'occupation']).size()
         .reset_index(name='weight')
         .sort_values('weight', ascending=False).head(10))

G = nx.DiGraph()
for _, row in top10.iterrows():
    G.add_edge(row['education'], row['occupation'], weight=row['weight'])

pos = nx.spring_layout(G, k=0.5, iterations=20)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=5000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
```

## Where It Appears

[[wk1-notebook]] section 3.3.1 (most common `education`↔`occupation` pairs). [[wk1-practice]]
extends the technique into **5 separate network clusters** (demographics, financial behaviour,
marketing channel, campaign intensity, core network with the target) for the Bank Marketing
dataset.

## Related Concepts

Different from **hierarchical clustering** (see
[[wk2-data-preparation-feature-engineering]]) which GROUPS levels of a categorical variable by a
continuous metric (e.g. high-income ratio) — the network graph here only VISUALISES co-occurrence
frequency, it does not cluster or decide which groups to merge.

## Wiki Links

[[wk1-notebook]] · [[wk1-data-understanding]] · [[wk1-practice]] ·
[[wk2-data-preparation-feature-engineering]]
