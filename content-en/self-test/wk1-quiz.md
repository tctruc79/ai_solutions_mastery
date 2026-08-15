---
type: self-test
title: "WK1 Self-Test — Data Understanding"
tags: [self-test, wk1, data-understanding]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Scope: [[wk1-data-understanding]], [[wk1-notebook]], [[wk1-practice]]. Practice questions
authored by Claude, not an official exam.

## Q1

`target_feature['income'].value_counts()` returns 4 distinct values: `<=50K`, `<=50K.`, `>50K`,
`>50K.`. Explain why there are 4 values instead of 2, and write one line of code to collapse them
into exactly 2 labels.

**Answer**: UCI Adult combines a train and a test portion; the test portion has a trailing `.`
while train doesn't → same meaning but counted as 4 distinct values. Fix:
`target_feature['income'].str.replace('.', '', regex=False).str.strip()` (see [[wk1-notebook]]
section 4.1).

## Q2

Why can't standard Pearson correlation be applied directly between `hours-per-week` and `income`
(text form)? What technique should be used instead, and what is the corresponding scipy function?

**Answer**: Pearson assumes both variables are continuous; `income` is categorical (even though
only 2 levels). You must binarise `income` (0/1) then use **point-biserial correlation** —
`scipy.stats.pointbiserialr()`. See [[point-biserial-correlation]].

## Q3

`input_features.isnull().sum()` reports 0 for all 14 columns, yet `workclass` actually has 1,836
missing rows. Why? Name a check that's more reliable than `isnull()` in this case.

**Answer**: Missing values are encoded with the character `'?'`, not standard `NaN`, so
`isnull()` doesn't catch them. A more reliable check:
`input_features['workclass'].value_counts()` — `'?'` will show up as a "category". See
[[wk1-data-understanding]], section "Missing Data Isn't Always NaN".

## Q4

In the assignment (Bank Marketing), the `poutcome` column is 81.7% missing. Why should this
column NOT be imputed the usual way?

**Answer**: This is **structural missingness** — clients are missing `poutcome` simply because
they were NEVER contacted in a previous campaign, not because of a data collection error. Correct
treatment: keep `'unknown'` / encode it as its own category (or a `no_previous_contact` boolean
flag), do not impute with mode/mean since that would erase exactly the "never contacted"
information — see [[wk1-practice]].

## Q5

Give one reason why fairness questions should be raised at the data-exploration stage rather than
waiting until a model exists.

**Answer**: Demographic skew in the raw data (e.g. Male 66.85%/Female 33.15%, `White` 85.5%) is
the ROOT CAUSE of later model bias; catching it earlier leaves more remediation options (collect
more balanced data, sample weighting, fairness constraints during training) — wait until after a
model is trained and the remaining options are mostly post-processing, far more limited. See
[[fairlearn]] (WK4) for remediation technique details.

## Wiki Links

[[wk1-data-understanding]] · [[wk1-notebook]] · [[wk1-practice]] · [[cumulative-quiz]]
