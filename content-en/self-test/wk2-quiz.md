---
type: self-test
title: "WK2 Self-Test — Data Preparation & Feature Engineering"
tags: [self-test, wk2, feature-engineering]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Scope: [[wk2-data-preparation-feature-engineering]], [[wk2-notebook]]. Practice questions
authored by Claude, not an official exam.

## Q1

`fnlwgt + age + education-num + hours-per-week + capital-gain` yields only 44,089 unique values
out of 48,842 rows. What is this problem called, and what's the correct fix?

**Answer**: This is a collision in a composite ID — two different rows can produce the same sum.
Fix: use `uuid.uuid4()` to generate a guaranteed-unique ID per row, rather than relying on a
column combination that only "looks" unique.

## Q2

Explain why `ColumnTransformer.fit()` should only be called on `Train_X`, never on `Test_X` or
the combined data.

**Answer**: If you `fit` on test (or combined data), the transformation parameters (e.g.
`StandardScaler`'s mean/std) will "see" test-set information — this is data leakage, making the
test-set evaluation score no longer reflect true performance on genuinely new data. See
[[sklearn-column-transformer-pipeline]].

## Q3

Why use hierarchical clustering to group `occupation` instead of one-hot encoding all 15 levels,
or manually grouping by intuition?

**Answer**: Full one-hot creates too many sparse columns, many behaving almost identically toward
the target; manually choosing groups by intuition isn't objective/reproducible. Clustering based
on each level's actual `>50K` rate yields OBJECTIVE, data-grounded groups that remain
interpretable (nameable). See [[hierarchical-clustering-categorical-grouping]].

## Q4

Random Forest and Logistic Regression both reach ~83-84% accuracy but F1 on the `>50K` class is
only 0.64. Why is this accuracy figure "less impressive than it looks"?

**Answer**: Because the `<=50K` class is 76.1% of the data — a DUMB model that always predicts
`<=50K` already achieves 76.1% accuracy without learning anything useful. 83-84% isn't far above
that floor, while F1 = 0.64 on the minority class shows the model is genuinely still weak at
identifying `>50K`. See [[wk2-data-preparation-feature-engineering]], section "High Accuracy Can
Be an Illusion".

## Wiki Links

[[wk2-data-preparation-feature-engineering]] · [[wk2-notebook]] · [[wk1-quiz]] ·
[[cumulative-quiz]]
