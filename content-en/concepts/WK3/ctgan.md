---
type: concept
title: "CTGAN (Conditional Tabular GAN)"
tags: [ctgan, gan, synthetic-data, imbalanced-data, deep-learning, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**CTGAN** (Conditional Tabular GAN) is a **Generative Adversarial Network (GAN)** architecture
designed specifically for tabular data — used to GENERATE synthetic data matching the real data
distribution, applied in the course to oversample the minority class for the Deep Learning
problem.

## Intuition / Why It Matters

[[imbalanced-data-resampling]] (SMOTE/ADASYN) only interpolates LINEARLY between existing real
data points — bounded within the current data's "convex hull", unable to create new points
OUTSIDE that range. CTGAN instead LEARNS a generator model representing the entire data
distribution through adversarial training — in principle it can produce more diverse and
realistic samples, capturing complex non-linear relationships across columns that linear
interpolation misses.

## Mechanism

The GAN architecture consists of 2 ADVERSARIALLY trained neural networks: the **Generator**
tries to produce fake samples that look like real data; the **Discriminator** tries to tell real
samples from fake ones — the two alternately improve each other (the Generator learns to fool
the Discriminator better, the Discriminator learns to detect better) until the Generator
produces sufficiently realistic data. Unlike typical image GANs, CTGAN handles continuous
columns separately (modelled via a Gaussian mixture) and categorical columns
(`discrete_columns=[...]`, using training-by-sampling to avoid mode collapse on imbalanced
columns). In the course: `CTGAN(epochs=5).fit(df_train, discrete_columns=['target'])` then
`ctgan.sample(n)` generates `n` new synthetic samples.

## Example

```python
import pandas as pd
from ctgan import CTGAN

df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
df_train['target'] = y_train

ctgan = CTGAN(epochs=5)                              # very few epochs in the original notebook
ctgan.fit(df_train, discrete_columns=['target'])

# Generate additional samples matching the current minority-class count
synthetic = ctgan.sample(len(y_train[y_train == 1]))
X_synthetic = synthetic.drop('target', axis=1).values
y_synthetic = synthetic['target'].values

import numpy as np
X_train_ctgan = np.vstack((X_train, X_synthetic))
y_train_ctgan = np.hstack((y_train, y_synthetic))
```

## Where It Appears

Appears at the end of [[wk3-notebook]]/"Deep Learning with Tensorflow/Keras" — comparing Base
Model vs **CTGAN Oversampling** vs Undersampling across a 5-fold `StratifiedKFold`. One
illustrative fold: CTGAN Oversampling (AUPRC ≈ 0.728) does NOT clearly outperform the Base Model
(≈ 0.732) — the full 5-fold output wasn't preserved in the source, so no firm conclusion can be
drawn yet. [[wk4-notebook]] expands CTGAN into one of 6 imbalance strategies across the full
9-model grid — per related pages, XGBoost + CTGAN Oversampling wins there (AUPRC ≈ 0.964 on CV,
but only ~0.80 on the real test set — an important lesson about correctly evaluating
synthetic-data-trained models).

## Related Concepts

Fundamentally different mechanism from [[imbalanced-data-resampling]] (SMOTE/ADASYN):
SMOTE/ADASYN is an INSTANT geometric computation (no training needed), while CTGAN is a NEURAL
NETWORK requiring training (even at just 5 epochs, still far slower, and needs enough data to
learn the distribution well). Trade-off: SMOTE/ADASYN are fast, simple, predictable in
behaviour; CTGAN is slower, more complex, but potentially captures complex data structure that
linear interpolation misses — though the course's empirical results do NOT show CTGAN always
beating SMOTE (compare [[wk3-notebook]]'s SMOTE+XGBoost AUPRC of 0.9590 against CTGAN+Deep
Learning's ~0.728 — though the two use different base models, so they aren't directly
comparable).

## Wiki Links

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[imbalanced-data-resampling]] ·
[[auprc]] · [[wk4-notebook]]
