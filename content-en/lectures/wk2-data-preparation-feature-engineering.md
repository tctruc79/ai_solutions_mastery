---
type: lecture
title: "WK2 — Data Preparation, Feature Engineering & Baseline Modelling"
tags: [data-preparation, feature-engineering, modelling, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb"
---

## Position in the Course

Follows directly from [[wk1-data-understanding]] — every Week-1 finding (disguised missing data,
`native-country` skew, new-feature ideas) now becomes concrete action: cleaning, feature
creation, and training the first two baseline models. The weak result (low F1 on the minority
class) leads straight into [[wk3-modelling-imbalance-deep-learning]].

## Learning Objectives

After this lesson you can: (1) build a complete sklearn `ColumnTransformer`+`Pipeline` from
scratch; (2) use hierarchical clustering to collapse a high-cardinality categorical variable into
a few meaningful groups; (3) explain why high accuracy (83-84%) can still be a BAD sign when data
is imbalanced; (4) read a Random Forest's `feature_importances_`.

## Core Theory

### 1. Data Preparation has 4 steps, not one vague "cleaning" step

CRISP-DM separates 4 distinct steps: **Integrate** (merge in other data sources, none this week)
→ **Select** (choose which rows/columns to keep, create a unique ID, split train/test) →
**Format & Clean** (normalise data types, handle anomalies) → **Construct** (create new
features). Separating these 4 explicitly prevents gaps — e.g. many people only think of "Clean"
and forget "Construct" is a SEPARATE, independent step.

### 2. Why a composite ID can be NOT unique

Trying an ID built from `fnlwgt + age + education-num + hours-per-week + capital-gain` yields
only 44,089 unique values across 48,842 rows — meaning there ARE collisions (two entirely
different rows can sum to the same total). Lesson: never assume a column combination that "looks"
unique actually is — always verify with `.nunique()` versus `.shape[0]`, and use `uuid.uuid4()`
when you need a guaranteed-unique ID.

### 3. Splitting train/test MUST stratify when the target is imbalanced

`train_test_split(..., stratify=Xy[TARGET_COL])` guarantees the `<=50K`/`>50K` ratio in train and
test EXACTLY MATCHES the original ratio (76.1%/23.9%) — without stratifying, a random split could
accidentally give the test set more/fewer minority examples than reality, skewing the evaluation
score.

### 4. Group high-cardinality categorical variables using data, not intuition

Instead of guessing how to bucket `occupation` (15 levels), the notebook COMPUTES the `>50K` rate
for every level then uses [[hierarchical-clustering-categorical-grouping]] to find levels with
similar behaviour — this approach is more objective and reproducible than manually choosing
groups by intuition.

### 5. Pipeline: packaging so a step is never "forgotten" on new data

[[sklearn-column-transformer-pipeline]] is the crux of the whole week — instead of writing loose
transformation statements (easy to forget re-applying to the test set), ALL feature-creation +
scaling logic is packaged into a single `Pipeline` object, `fit` once on train, `transform`ed
identically on any new data (the test set, or real data at deployment time in [[wk4-notebook]]).

### 6. High accuracy can be an illusion when data is imbalanced

Both Random Forest and Logistic Regression reach 83-84% accuracy — sounds good, but F1 on the
`>50K` class is only 0.64. Since `<=50K` is 76% of the data, a DUMB model that always predicts
"`<=50K`" already achieves 76% accuracy while learning nothing — the 83-84% figure isn't that far
above that floor. This is why WK3 switches entirely to measuring with AUPRC.

## Worked Example

```python
import uuid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Select — unique id + stratified split
Xy['uuid'] = [uuid.uuid4() for _ in range(len(Xy))]
Train_Xy, Test_Xy = train_test_split(
    Xy, test_size=0.2, stratify=Xy['income'], random_state=42
)

# Feature Engineering — reuse the pipeline from [[sklearn-column-transformer-pipeline]]
Train_X_transformed = feature_engineering_pipeline.fit_transform(Train_X)
Test_X_transformed = feature_engineering_pipeline.transform(Test_X)   # do NOT refit

# Modelling — baseline
rf_model = RandomForestClassifier(random_state=42).fit(Train_X_transformed, Train_y_transformed)
print(classification_report(Test_y_transformed, rf_model.predict(Test_X_transformed)))
# accuracy 0.83, but F1 on the '>50K' class is only 0.64 -> class imbalance needs handling (WK3)
```

## Pitfalls & Practical Notes

- **Running `fit_transform` on ALL data (train+test combined) before splitting** — leaks test
  information into the transformation parameters (e.g. `StandardScaler` learning mean/std that
  include test data); ALWAYS split first, `fit` only on train, `transform` (without fitting) on
  test.
- **Forgetting `stratify=` when the target is imbalanced** — leads to different class ratios
  between train/test, making the evaluation score unreliable.
- **Only looking at accuracy while ignoring per-class F1/precision/recall** — on imbalanced data,
  accuracy almost always "looks fine" in a misleading way.

## Related Concepts

[[sklearn-column-transformer-pipeline]] · [[hierarchical-clustering-categorical-grouping]] ·
[[random-forest]] · [[logistic-regression]]

## Quick Self-Check

1. Why must you `stratify=` when splitting train/test for this problem? What happens if you skip
   it?
2. Which set should `ColumnTransformer.fit()` be called on — train, test, or both combined? Why?
3. 83% accuracy sounds good, but why isn't it enough to conclude the model is good for this
   problem?

Full question bank: [[wk2-quiz]].

## Wiki Links

[[wk2-notebook]] · [[wk2-practice]] · [[wk1-data-understanding]] ·
[[wk3-modelling-imbalance-deep-learning]] · [[sklearn-column-transformer-pipeline]] ·
[[hierarchical-clustering-categorical-grouping]] · [[random-forest]] · [[logistic-regression]] ·
[[overview]]
