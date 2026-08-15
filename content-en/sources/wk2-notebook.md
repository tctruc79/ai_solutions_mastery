---
type: source
title: "WK2 Notebook — Data Preparation, Feature Engineering & Modelling"
tags: [data-preparation, feature-engineering, modelling, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb"
---

## Source Info

File `honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb`, 148 cells.
Libraries NEW versus Week 1: `sklearn.model_selection`, `sklearn.compose.ColumnTransformer`,
`sklearn.pipeline.Pipeline`, `sklearn.preprocessing`, `sklearn.ensemble.RandomForestClassifier`,
`sklearn.linear_model.LogisticRegression`, `sklearn.metrics`, `plotly`, `uuid`.

## Learning Objectives

4 major blocks matching the original headings: **Data Understanding in Short** (recap Week 1 via
reusable functions) → **Data Preparation** (Integrate → Select → Format & Clean → Construct Data)
→ **Feature Engineering** (Generate Test/Train → Feature Creation/Transformation/Encoding/Scaling)
→ **Modelling** (Select Technique → Build → Assess → Cross Validation → Feature Selection).

## Notebook Structure

- **Data Understanding in Short** — 3 NEW utility functions replacing Week-1's scattered code:
  `load_data_from_uci()`, `understand_data()`, `clean_y()`, `make_x_y()`.
- **1. Data Preparation** → 2.1 Create Unique ID (`uuid`), 2.2 Create Test Set (stratified
  split), 3. Format & Clean Data (deep dive per column: workclass, education, marital-status,
  occupation with hierarchical clustering, relationship, race, sex, capital-gain/loss with log
  transform, hours-per-week, native-country → decision to filter `== 'United-States'`), 4. Construct
  Data (new engineered flags).
- **2. Feature Engineering** — `CustomFeatureCreator` (sklearn `TransformerMixin`), Column
  Transformer (`StandardScaler` for continuous features, `passthrough` for binary flags, `drop`
  for raw categoricals), `Pipeline` combining both steps.
- **3. Modelling** — 1. Select Technique (rationale for Random Forest), 2. Build Model
  (LogisticRegression + RandomForestClassifier), 3. Assess (classification_report, confusion
  matrix), 4. Cross Validation (10-fold), 5. Feature Selection (feature_importances_).
- **"Next Week" preview** — ROC curve, Precision-Recall curve (sets up AUPRC for WK3).

## Technical Walkthrough

### Data Preparation — decision to filter to US only

After exploring `native-country` (US is 90.2%, `>50K` rate differs sharply by country — USA
~24.4% vs Mexico ~5.3%), the notebook decides to **keep only `native-country ==
'United-States'`** for the entire modelling phase — avoiding mixing several different income
distributions into one model.

### Hierarchical clustering to group high-cardinality categorical variables

`occupation` has 15 levels — instead of one-hot encoding all 15, the notebook computes each
occupation's `>50K` rate then uses `scipy.cluster.hierarchy.linkage(method='ward')` to cluster
occupations with similar income behaviour, yielding 7 meaningful groups (`occupation_cleaner`,
`occupation_repair`, `occupation_admin`...) instead of 15 disjoint columns.

### Feature Engineering — a pipeline reused in every subsequent week

`CustomFeatureCreator` (subclasses `BaseEstimator, TransformerMixin`) creates 0/1 flags
(`workclass_gov`, `is_married`, `race_white`...) plus `log_capital_gain =
log1p(capital-gain)`. `ColumnTransformer` then: `StandardScaler` on
`age`/`hours-per-week`/`log_capital_gain`, drops every raw categorical column, `passthrough` for
the binary flags + `education-num`. All wrapped in a single `Pipeline` — **this exact pipeline is
reused UNCHANGED by WK3 and WK4**, see [[sklearn-column-transformer-pipeline]].

### Modelling — baseline Logistic Regression & Random Forest

`log_reg_model = LogisticRegression(random_state=42, max_iter=1000)` and `rf_model =
RandomForestClassifier(random_state=42)`, both fit directly on `Train_X_transformed`. Evaluated
via `classification_report` + confusion matrix + `cross_val_score(cv=10,
scoring='f1_weighted')`.

## Key Numbers

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy (test set) | 0.83 | 0.84 |
| F1 on `>50K` class | 0.64 | 0.64 |
| 10-fold CV f1_weighted (mean) | 0.8298 | 0.8319 |
| Number of features after feature engineering | 19 | 19 |
| Train/Test size (US only) | 35,065 / 8,767 | — |

## Quirks / Issues Found

Both models reach ~83-84% accuracy but only 0.64 F1 on the minority `>50K` class — a clear sign
of class imbalance (~76/24) not yet addressed this week (accuracy "looks fine" but is
misleading). This is the direct motivation for WK3 switching entirely to measuring with **AUPRC**
instead of plain accuracy/F1, see [[wk3-notebook]].

## Wiki Links

[[wk2-data-preparation-feature-engineering]] · [[wk2-practice]] · [[wk1-notebook]] ·
[[wk3-notebook]] · [[sklearn-column-transformer-pipeline]] ·
[[hierarchical-clustering-categorical-grouping]] · [[random-forest]] · [[overview]]
