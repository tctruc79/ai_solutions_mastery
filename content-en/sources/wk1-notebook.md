---
type: source
title: "WK1 Notebook — Data Understanding (honeyg_ai_practitioner_wk1_Data_Understanding.ipynb)"
tags: [data-understanding, crisp-dm, eda, uci-adult]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai_practitioner_wk1_Data_Understanding.ipynb"
---

## Source Info

File `honeyg_ai_practitioner_wk1_Data_Understanding.ipynb`, 98 cells (49 markdown + 49 code),
kernel `AIDS_Maryam_Miradi` (Python 3.13).

This is the Week-1 notebook of [[maryam-miradi]], opening the AI Practitioner programme with the
**Data Understanding** phase of CRISP-DM.

## Learning Objectives

The notebook declares 4 sections that map directly onto CRISP-DM: Collect Initial Data → Describe
Data → Explore Data → Verify Data Quality.

## Notebook Structure

- **Setup and Installation** — install `ucimlrepo`, `networkx`.
- **1. Collect Initial Data** — 1.1 Census Income Data, loaded via `fetch_ucirepo(id=2)`.
- **2. Describe Data** — 2.1 Data Structure (shape, `.head()`), 2.2 Metadata (`adult.metadata`,
  the original UCI `.names` file).
- **3. Explore Data** — 3.1 General Exploration (`.info()`, `.describe()`, category percentage
  breakdowns), 3.2 Correlations (Pearson heatmap for numeric features + Point-Biserial for
  numeric↔binary), 3.3 Advanced Exploration (education↔occupation network graph).
- **4. Verify Data Quality** — 4.1 Clean Data (normalise the `income` label), 4.2 Impute Missing
  Values (principle only, not yet executed), 4.3 Ideas for Feature Generation / Observations
  (fairness notes).

## Libraries & Tools

`sys`, `packaging.version`, `requests`, `pprint`, `json` (general utilities); `numpy`, `pandas`
(data); `seaborn`, `matplotlib` (visualisation); `scipy.stats.pointbiserialr` (statistics);
`networkx` (network analysis); `ucimlrepo.fetch_ucirepo` (data loading); `sklearn` (version check
only, no model built this week).

## Technical Walkthrough

### 1. Collect & Describe

`fetch_ucirepo(id=2)` returns `adult.data.features` (48,842 rows × 14 columns) and
`adult.data.targets` (48,842 × 1, column `income`). The original UCI metadata (`adult.metadata`)
and the `.names` description file are fetched with a hand-written `download_file()` helper for
cross-checking.

### 2. Explore — descriptive statistics and data imbalance

`input_features.isnull().sum()` shows `workclass` (963 rows) and `occupation` (966 rows) with
blanks — actually the character `'?'`, not standard `NaN`. The `income` column initially has
**4** seemingly distinct labels: `<=50K`, `<=50K.`, `>50K`, `>50K.` — a mix of UCI's train
portion (no trailing dot) and test portion (trailing dot). Gender is heavily skewed: Male 66.85%
/ Female 33.15%; race skews toward `White` at 85.50%.

### 3. Correlation — two techniques for two variable types

For numeric ↔ numeric columns: a standard Pearson correlation heatmap (`.corr()` +
`sns.heatmap`). For numeric ↔ binary target (`income`): **point-biserial correlation** via
`scipy.stats.pointbiserialr` — see [[point-biserial-correlation]]. Result: `education-num` (0.26)
and `age` (0.18) correlate most strongly (positively) with high income.

### 3.3 Network Analysis — visualising categorical relationships

Uses `networkx.DiGraph` to draw the top-10 most common `education → occupation` pairs (counted
via `groupby(['education','occupation']).size()`), see [[network-graph-categorical-eda]].

### 4. Verify Data Quality

`target_feature['income'].str.replace('.', '', regex=False).str.strip()` collapses the 4 labels
into exactly 2: `<=50K` (37,155) / `>50K` (11,687). The notebook emphasises a principle: **never
auto-fill missing values with 0/mean without investigating first** — section 4.2 states the
principle only, no actual imputation is performed yet.

## Key Numbers

| Metric | Value |
|---|---|
| Original rows / columns | 48,842 × 14 (input) + 1 (target) |
| `income` after cleaning | `<=50K` 37,155 (76.1%) · `>50K` 11,687 (23.9%) |
| `workclass` missing (`'?'`) | 1,836 (3.76%) |
| `occupation` missing (`'?'`) | 1,843 (~3.8%, after merge) |
| Strongest point-biserial correlation with `income` | `education-num` r≈0.26, `age` r≈0.18 |
| `capital-gain` > 0 | 4,035 / 48,842 rows (8.3%) |

## Quirks / Issues Found

The notebook loads `input_features`/`target_feature` **separately** across several cells (cell 68
cleans `all_features['income']` locally, cell 93 re-cleans `target_feature` — two similar
operations not sharing one function) — this is exactly why Week 2 extracts this logic into a
reusable `clean_y()` function, see [[wk2-notebook]].

## Wiki Links

[[wk1-data-understanding]] · [[wk1-practice]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[wk2-notebook]] · [[overview]]
