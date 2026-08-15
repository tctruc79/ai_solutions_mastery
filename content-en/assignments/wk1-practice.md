---
type: assignment
title: "WK1 Assignment — Data Understanding (Bank Marketing Dataset)"
tags: [assignment, data-understanding, bank-marketing, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Assignments/honeyg_Assignment_wk1_Data_Understanding.ipynb"
also_covers:
  - "raw/Assignments/Tran Chanh Truc_wk1_assessment_feedback_Dr_Maryam_Miradi.pdf"
---

## Assignment Brief

Apply the exact Data Understanding methodology from [[wk1-data-understanding]] (Collect →
Describe → Explore → Verify Data Quality per [[crisp-dm]]) to a **different dataset** chosen by
the student — not a repeat of the lecture's UCI Adult dataset, testing the ability to generalise
the process to a new problem.

Trực chose the **UCI Bank Marketing Dataset** (id=222) — predicting whether a client subscribes
to a term deposit (`y` = yes/no) based on phone-marketing campaign data from a Portuguese bank
(May 2008 – Nov 2010).

## What Was Done

A 448-cell notebook — much larger than the original lecture (98 cells) because the Bank Marketing
archive is more complex (zip nested inside zip, 4 file variants: `bank.csv`/`bank-full.csv`/
`bank-additional.csv`/`bank-additional-full.csv`) and Trực extended the analysis well beyond the
minimum requirement.

Highlights beyond the original lecture:

- **Dual-source loading** — a customised `download_file()` handling the nested-zip extraction,
  alongside `fetch_ucirepo(id=222)`, cross-checked against each other for consistency.
- **Reusable function library** — `categorical_summary()`, `missing_value_summary()`,
  `plot_numeric_distributions()`, `group_numeric_summary_flat()`,
  `group_categorical_summary_flat()`: single-responsibility, docstring'd, reused throughout
  instead of copy-pasted.
- **Data-leakage detection** — proactively flagging `duration` (call length) as a post-hoc
  variable: only known AFTER the call ends, so it cannot be used to predict BEFORE calling — an
  ML-awareness insight the brief did not explicitly require.
- **63 bivariate violin plots** across 7 numeric × 9 categorical features, and **5 network-graph
  clusters** (demographics, financial behaviour, marketing channel, campaign intensity, core
  target network) — extending [[network-graph-categorical-eda]] well beyond the single variable
  pair in the original lecture.
- **Data-quality summary table (section 4.6)** — 8 issues listed systematically: severity,
  affected data, proposed fix (e.g. leaking `duration` → excluded from the pre-call prediction
  model; `poutcome` missing 81.7% → kept as its own category rather than imputed).

## Comparison with the Lecture

Follows the same [[crisp-dm]] 4-step process and reuses two Week-1 core techniques as-is:
[[point-biserial-correlation]] (section 3.2.4.2, comparing `duration`/`pdays`/`previous`/
`campaign` against target `y`) and [[network-graph-categorical-eda]] (extended to 5 clusters
instead of one pair). The biggest difference: the lecture has a single data source (UCI Adult),
while this assignment must handle 4 files across 2 different schemas (the `bank`/`bank-full`
family, 16 columns, vs the `bank-additional`/`bank-additional-full` family, 20 columns, adding 5
macroeconomic indicators) — requiring an extra schema-consistency verification step (section 4.5)
that the original lecture didn't need with only one source.

## Feedback from Dr. Maryam Miradi

**Score: 92/100 — Strong pass, A-.**

| Criterion | Score |
|---|---|
| Data collection & loading | 95% |
| Data description & metadata | 92% |
| Exploratory analysis depth | 90% |
| Visualisation quality | 88% |
| Correlation analysis | 93% |
| Data quality verification | 82% |
| Findings & insights report | 95% |
| Code quality & reusability | 85% |

Overall comment: "Excellent depth and reusable function architecture throughout... Two bugs in
section 4 prevent a top band score but do not diminish the quality of the analytical work."

**2 bugs flagged:**

1. In the `day` → `day_of_week` rename step, 3 of the 4 initial rename calls were chained off the
   wrong variable — all three continued from `bank_full_input_features` instead of their own
   variable, silently OVERWRITING `bank_input_features`, `bank_additional_full_input_features`,
   and `bank_additional_input_features` with `bank-full.csv` data — more dangerous because
   `bank-additional-full` is an ENTIRELY DIFFERENT schema (adds 5 macroeconomic variables), so
   this bug silently loses a whole feature set with no error raised.
2. Section 4.3 "Feature generation / Observations" still contained copy-pasted content from the
   UCI Adult lecture (referencing `workclass`, `Capital_Gain`, gender/race fairness statistics, US
   Census links) — none of these fields exist in the Bank Marketing dataset; the section needed
   rewriting for the actual dataset.

**Current status (verified 2026-08-14):** the notebook in `raw/Assignments/` HAS fixed both
issues. The rename section now uses the correct loop pattern over `datasets =
[bank_full_input_features, bank_input_features, bank_additional_full_input_features,
bank_additional_input_features]` + `for df in datasets: df.rename(...)` (matching the feedback's
DRY recommendation), followed by a verification cell confirming all 4 datasets retain their
original shape. The original section 4.3 has been replaced with a proper 4.3–4.6 sequence
matching Bank Marketing content (Duplicate Records, Validity & Range Checks, Cross-Dataset
Structural Consistency, Data Quality Verification Summary).

Other recommendations from the feedback also show signs of being applied in the current version:
imports are consolidated into a single "Load Libraries" cell (including `sklearn`,
`fetch_ucirepo` — previously split apart), and `download_file()` carries inline comments
explaining WHY (e.g. why `.endswith('.zip')` is checked twice — due to this dataset's nested-zip
structure).

## Action Items for Next Week

- Apply the **DRY-via-loop** habit (learned directly from this week's rename bug) to similarly
  repeated operations in Week 2 (feature engineering across multiple columns at once) — verify
  immediately with `.info()`/`.shape` after each loop instead of blindly trusting it.
- Keep up the "Analysis → Observation" loop right after every output-producing cell — the current
  version does this fairly consistently in sections 2–3, keep the habit as the wk2 notebook grows
  more complex (many chained data-transformation steps).
- When wk2/wk3/wk4 assignments are ingested later, also check whether the lessons learned here
  (DRY loops, consolidated imports, WHY-comments) are sustained throughout.

## Wiki Links

[[wk1-data-understanding]] · [[wk1-notebook]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[maryam-miradi]]
