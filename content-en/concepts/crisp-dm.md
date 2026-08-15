---
type: concept
title: "CRISP-DM"
tags: [methodology, crisp-dm, workflow]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Definition

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is a 6-phase process for
structuring a data-science project end-to-end.

All 4 weeks of the course follow this exact order — each lecture notebook uses markdown headings
that literally match the CRISP-DM phase names.

## Intuition / Why It Matters

Jumping straight to model training while skipping the data-understanding and cleaning steps means
the model faithfully learns whatever errors/biases already exist in the data (garbage in, garbage
out).

## The 6 Phases

1. **Business Understanding** — understand the business goal/problem (the course doesn't dedicate
   a separate large section to this, but always states it briefly at the top of each notebook,
   e.g. the >\$50K income prediction task).
2. **Data Understanding** = this course's WK1 — Collect → Describe → Explore → Verify Data
   Quality, see [[wk1-data-understanding]].
3. **Data Preparation** = the first half of WK2 — Integrate → Select → Format & Clean → Construct
   Data, see [[wk2-data-preparation-feature-engineering]].
4. **Modelling** = the second half of WK2 + all of WK3 + the first half of WK4 — select
   technique, build model, assess, feature selection.
5. **Evaluation** — folded into WK4's Model Selection step in this course (re-checking on the
   REAL test set rather than trusting cross-validation numbers computed on resampled data) — see
   [[wk4-model-selection-explainability-fairness-deployment]].
6. **Deployment** = the end of WK4 — persisting the model with `joblib`, a Streamlit app
   scaffold.

## Example

```python
# The Data Understanding phase (WK1) always starts with shape + describe, no model building yet
input_features.shape        # (48842, 14)
input_features.describe()   # descriptive statistics before any processing decision is made
```

## Where It Appears

All 4 weeks: [[wk1-notebook]], [[wk2-notebook]], [[wk3-notebook]], [[wk4-notebook]].

## Related Frameworks

CRISP-DM differs from the **MLOps lifecycle** (adds CI/CD, continuous post-deployment monitoring
— this course only reaches a Streamlit scaffold, not deep MLOps) and from **OSEMN**
(Obtain-Scrub-Explore-Model-iNterpret, a leaner variant that de-emphasises Business
Understanding).

## Wiki Links

[[overview]] · [[wk1-data-understanding]] · [[wk2-data-preparation-feature-engineering]] ·
[[wk4-model-selection-explainability-fairness-deployment]]
