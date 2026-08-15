---
type: lecture
title: "WK1 — Data Understanding"
tags: [data-understanding, crisp-dm, eda, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai_practitioner_wk1_Data_Understanding.ipynb"
---

## Position in the Course

This is the first week, the **Data Understanding** phase of [[crisp-dm]] — before touching any
model, you must understand what the data has, what's missing, where it's skewed. Week 2 directly
reuses every finding here (especially the decision to filter to
`native-country == 'United-States'` and the new-feature ideas) to prepare the data for modelling.

## Learning Objectives

After this lesson you can: (1) load data from the UCI Machine Learning Repository via
`ucimlrepo`; (2) tell when to use ordinary Pearson correlation versus point-biserial correlation;
(3) recognise "disguised" missing data (not always literal `NaN`); (4) ask fairness questions at
the data-exploration stage already, before any model exists.

## Core Theory

### 1. Why Data Understanding comes before Modelling

A model is only as good as the data it's trained on. Skip the data-understanding step and
problems like oddly-encoded missing values (`'?'` instead of `NaN`), inconsistent target labels
(`>50K` vs `>50K.`), or demographic skew silently leak into the model and get mistaken for a real
signal instead of a data defect.

### 2. Loading data from UCI via `ucimlrepo`

Standard approach: go to the UCI Machine Learning Repository page for the dataset, click "Import
in Python", copy the generated code.

```python
from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)              # id=2 is the Adult Census Income dataset's own code
input_features = adult.data.features     # DataFrame 48,842 × 14
target_feature = adult.data.targets      # DataFrame 48,842 × 1, column 'income'
print(adult.metadata['abstract'])        # problem description straight from the source
```

### 3. Two kinds of correlation for two kinds of problem

When X is continuous and Y is also continuous → ordinary Pearson (`df.corr()`). When X is
continuous but Y is a BINARY target (like `income`) → **point-biserial** (see
[[point-biserial-correlation]]), do NOT run Pearson directly on a string label.

```python
from scipy.stats import pointbiserialr

y_binary = (target_feature['income'] == '>50K').astype(int)
r, _ = pointbiserialr(y_binary, input_features['age'])   # r ≈ 0.18
```

### 4. Missing data isn't always `NaN`

`input_features.isnull().sum()` reports 0 for `workclass`, but the column actually HAS 1,836
missing rows — they're just encoded as the character `'?'`. This is the week's most important
lesson: ALWAYS inspect `value_counts()` on categorical columns before trusting `isnull()`.

```python
input_features['workclass'].value_counts()
# ...
# ?                    1836      <- this is actually the missing value, not a real category
# Never-worked            10
```

### 5. The "no blind imputation" principle

The original notebook states clearly: never auto-fill missing values with 0/mean without first
investigating the MECHANISM of missingness — data can be missing **structurally** (missing by
logic, e.g. `poutcome='unknown'` when a client was never previously contacted — see
[[wk1-practice]]) or genuinely at random. The correct treatment differs completely between the
two cases.

### 6. Asking fairness questions RIGHT FROM the data-exploration stage

Before training any model, the notebook already notes: Male 66.85% / Female 33.15%, and `White`
at 85.50% — a meaningful skew relative to actual US population demographics. This is an early
warning sign: a model trained on this data COULD learn gender/race bias, and the fairness
question should be raised now, not after a model already exists (this topic returns in full in
[[fairlearn]], Week 4).

## Worked Example

The minimal sequence from raw data load to a clean target, ready for Week 2:

```python
from ucimlrepo import fetch_ucirepo
from scipy.stats import pointbiserialr
import pandas as pd

# 1. Collect
adult = fetch_ucirepo(id=2)
X, y = adult.data.features, adult.data.targets

# 2. Describe
print(X.shape, y.shape)                          # (48842, 14) (48842, 1)
print(X.isnull().sum())                           # careful: '?' does NOT show up here

# 3. Explore — point-biserial correlation, not ordinary Pearson
y_bin = (y['income'].str.contains('>50K')).astype(int)
for col in ['age', 'education-num', 'hours-per-week']:
    r, _ = pointbiserialr(y_bin, X[col])
    print(col, round(r, 3))

# 4. Verify Data Quality — normalise the target to exactly 2 labels
y['income'] = y['income'].str.replace('.', '', regex=False).str.strip()
print(y['income'].value_counts())                 # exactly 2 labels: <=50K / >50K
```

## Pitfalls & Practical Notes

- **Trusting `isnull().sum() == 0` without checking `value_counts()`** — real-world data often
  encodes missingness with a special token (`'?'`, `'unknown'`, `-1`, `999`...) depending on the
  source.
- **Running Pearson on a string column** — will error or produce meaningless results; binarise
  the target first, then use point-biserial.
- **Skipping fairness because "there's no model yet"** — the earlier the better, since skew in
  the raw data is the root cause of later model bias; catching it early leaves more remediation
  options (collect more data, sample weighting, fairness constraints).

## Related Concepts

[[crisp-dm]] · [[point-biserial-correlation]] · [[network-graph-categorical-eda]]

## Quick Self-Check

1. Why can't you run ordinary Pearson correlation between `age` and the `income` column (text
   `<=50K`/`>50K`) without any preprocessing?
2. `input_features.isnull().sum()` reports 0 for `workclass` — does that mean the column has no
   missing data at all? Why or why not?
3. Give an example of **structural** missingness (missing by logic) as opposed to random
   missingness.

Full question bank: [[wk1-quiz]].

## Wiki Links

[[wk1-notebook]] · [[wk1-practice]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[wk2-data-preparation-feature-engineering]] · [[overview]]
