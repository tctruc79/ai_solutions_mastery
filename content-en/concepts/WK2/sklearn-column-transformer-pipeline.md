---
type: concept
title: "ColumnTransformer + Pipeline Pattern"
tags: [sklearn, feature-engineering, pipeline, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

A sklearn design pattern that bundles the entire feature-transformation step (creating new
features, scaling, encoding) into **a single object**, guaranteeing the exact same transformation
is applied to the training set and to the test set/new data.

Composed of 3 nested pieces: a **custom transformer** (subclassing `BaseEstimator,
TransformerMixin`) to create new features → **`ColumnTransformer`** to apply different transforms
to different column groups → **`Pipeline`** to chain the steps into one `fit`/`transform` flow.

## Intuition / Why It Matters

If feature-transformation code is written as loose, unpackaged statements, it's easy to forget to
apply the exact same transform to the test set (e.g. forgetting `log1p` for `capital-gain` at
test time), or to accidentally leak information from test into train by fitting a scaler on the
whole dataset instead of only on train. Bundling into one pipeline eliminates both risks by
design.

## Mechanism

`feature_engineering_pipeline.fit(Train_X)` learns parameters (e.g. `StandardScaler`'s mean/std)
ONLY from the TRAIN SET. `.transform(Test_X)` then reapplies those exact learned parameters to
test — the test set never influences the transformation parameters back.

## Example

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class CustomFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self                      # learns nothing extra, just creates columns with a fixed formula
    def transform(self, X):
        X = X.copy()
        X['is_married'] = (X['marital-status'] == 'Married-civ-spouse').astype(int)
        X['log_capital_gain'] = np.log1p(X['capital-gain'])
        return X

feature_transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'hours-per-week', 'log_capital_gain']),
    ('cat', 'drop', ['workclass', 'education', 'marital-status', ...]),
], remainder='passthrough')              # keep the 0/1 flags already created

pipeline = Pipeline([
    ('feature_creation', CustomFeatureCreator()),
    ('feature_transform', feature_transformer),
])
pipeline.fit(Train_X)                    # learns parameters ONLY from train
Train_X_transformed = pipeline.transform(Train_X)
Test_X_transformed = pipeline.transform(Test_X)   # reapplied, NOT refit
```

## Where It Appears

Introduced in [[wk2-notebook]], **reused unchanged** in [[wk3-notebook]] and [[wk4-notebook]] —
the same `build_features()` function appears across 3 weeks, only how the output is used
downstream differs (adding resampling, different models...).

## Related Concepts

Different from `OneHotEncoder` applied directly to all categorical columns (not used in this
pipeline — raw categorical columns are `drop`ped after their information has been manually
"compressed" into 0/1 flags via `CustomFeatureCreator`, a deliberate design choice over automatic
full one-hot encoding, see [[hierarchical-clustering-categorical-grouping]] to understand why
manual grouping was chosen over one-hot).

## Wiki Links

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] ·
[[hierarchical-clustering-categorical-grouping]] · [[wk3-notebook]] · [[wk4-notebook]]
