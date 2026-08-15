---
type: source
title: "WK2 Notebook — Data Preparation, Feature Engineering & Modelling"
tags: [data-preparation, feature-engineering, modelling, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb"
---

## Thông tin nguồn - <span class="en">Source Info</span>

File `honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb`, 148 cell.
Thư viện MỚI so với tuần 1: `sklearn.model_selection`, `sklearn.compose.ColumnTransformer`,
`sklearn.pipeline.Pipeline`, `sklearn.preprocessing`, `sklearn.ensemble.RandomForestClassifier`,
`sklearn.linear_model.LogisticRegression`, `sklearn.metrics`, `plotly`, `uuid`.
<br><span class="en">File `honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_
Modelling.ipynb`, 148 cells. Libraries NEW versus Week 1: `sklearn.model_selection`,
`sklearn.compose.ColumnTransformer`, `sklearn.pipeline.Pipeline`, `sklearn.preprocessing`,
`sklearn.ensemble.RandomForestClassifier`, `sklearn.linear_model.LogisticRegression`,
`sklearn.metrics`, `plotly`, `uuid`.</span>

## Mục tiêu buổi học - <span class="en">Learning Objectives</span>

3 khối lớn theo đúng heading gốc: **Data Understanding in Short** (tóm lược lại WK1 bằng hàm tái
sử dụng) → **Data Preparation** (Integrate → Select → Format & Clean → Construct Data) →
**Feature Engineering** (Generate Test/Train → Feature Creation/Transformation/Encoding/Scaling)
→ **Modelling** (Select Technique → Build → Assess → Cross Validation → Feature Selection).
<br><span class="en">4 major blocks matching the original headings: **Data Understanding in
Short** (recap Week 1 via reusable functions) → **Data Preparation** (Integrate → Select →
Format & Clean → Construct Data) → **Feature Engineering** (Generate Test/Train → Feature
Creation/Transformation/Encoding/Scaling) → **Modelling** (Select Technique → Build → Assess →
Cross Validation → Feature Selection).</span>

## Cấu trúc notebook - <span class="en">Notebook Structure</span>

- **Data Understanding in Short** — 3 hàm tiện ích MỚI thay cho code rời rạc ở WK1:
  `load_data_from_uci()`, `understand_data()`, `clean_y()`, `make_x_y()`.
  <br><span class="en">**Data Understanding in Short** — 3 NEW utility functions replacing
  Week-1's scattered code: `load_data_from_uci()`, `understand_data()`, `clean_y()`,
  `make_x_y()`.</span>
- **1. Data Preparation** → 2.1 Create Unique ID (`uuid`), 2.2 Create Test Set (stratified
  split), 3. Format & Clean Data (đi sâu từng cột: workclass, education, marital-status,
  occupation với hierarchical clustering, relationship, race, sex, capital-gain/loss với log
  transform, hours-per-week, native-country → quyết định lọc `== 'United-States'`), 4. Construct
  Data (tạo các cờ đặc trưng mới).
  <br><span class="en">**1. Data Preparation** → 2.1 Create Unique ID (`uuid`), 2.2 Create Test
  Set (stratified split), 3. Format & Clean Data (deep dive per column: workclass, education,
  marital-status, occupation with hierarchical clustering, relationship, race, sex,
  capital-gain/loss with log transform, hours-per-week, native-country → decision to filter
  `== 'United-States'`), 4. Construct Data (new engineered flags).</span>
- **2. Feature Engineering** — `CustomFeatureCreator` (sklearn `TransformerMixin`), Column
  Transformer (`StandardScaler` cho biến liên tục, `passthrough` cho cờ nhị phân, `drop` cho
  biến phân loại thô), `Pipeline` gộp cả 2 bước.
  <br><span class="en">**2. Feature Engineering** — `CustomFeatureCreator` (sklearn
  `TransformerMixin`), Column Transformer (`StandardScaler` for continuous features,
  `passthrough` for binary flags, `drop` for raw categoricals), `Pipeline` combining both
  steps.</span>
- **3. Modelling** — 1. Select Technique (lý do chọn Random Forest), 2. Build Model
  (LogisticRegression + RandomForestClassifier), 3. Assess (classification_report, confusion
  matrix), 4. Cross Validation (10-fold), 5. Feature Selection (feature_importances_).
  <br><span class="en">**3. Modelling** — 1. Select Technique (rationale for Random Forest), 2.
  Build Model (LogisticRegression + RandomForestClassifier), 3. Assess (classification_report,
  confusion matrix), 4. Cross Validation (10-fold), 5. Feature Selection
  (feature_importances_).</span>
- **"Next Week" preview** — ROC curve, Precision-Recall curve (đặt nền cho AUPRC ở WK3).
  <br><span class="en">**"Next Week" preview** — ROC curve, Precision-Recall curve (sets up
  AUPRC for WK3).</span>

## Walkthrough kỹ thuật theo từng phần - <span class="en">Technical Walkthrough</span>

### Data Preparation — quyết định lọc chỉ US

Sau khi khám phá `native-country` (US chiếm 90.2%, tỷ lệ `>50K` khác biệt rõ theo quốc gia — Mỹ
~24.4% vs Mexico ~5.3%), notebook quyết định **chỉ giữ lại `native-country == 'United-States'`**
cho toàn bộ phần modelling — tránh trộn lẫn nhiều phân phối thu nhập khác nhau vào 1 mô hình.
<br><span class="en">After exploring `native-country` (US is 90.2%, `>50K` rate differs sharply
by country — USA ~24.4% vs Mexico ~5.3%), the notebook decides to **keep only
`native-country == 'United-States'`** for the entire modelling phase — avoiding mixing several
different income distributions into one model.</span>

### Hierarchical clustering để nhóm biến phân loại nhiều mức

`occupation` có 15 mức — thay vì one-hot 15 cột, notebook tính tỷ lệ `>50K` cho từng occupation
rồi dùng `scipy.cluster.hierarchy.linkage(method='ward')` để phân cụm các occupation có hành vi
thu nhập tương tự nhau, từ đó chọn ra 7 nhóm nghĩa (`occupation_cleaner`, `occupation_repair`,
`occupation_admin`...) thay vì giữ nguyên 15 cột rời rạc.
<br><span class="en">`occupation` has 15 levels — instead of one-hot encoding all 15, the
notebook computes each occupation's `>50K` rate then uses
`scipy.cluster.hierarchy.linkage(method='ward')` to cluster occupations with similar income
behaviour, yielding 7 meaningful groups (`occupation_cleaner`, `occupation_repair`,
`occupation_admin`...) instead of 15 disjoint columns.</span>

### Feature Engineering — pipeline tái sử dụng cho mọi tuần sau

`CustomFeatureCreator` (kế thừa `BaseEstimator, TransformerMixin`) tạo các cờ 0/1
(`workclass_gov`, `is_married`, `race_white`...) + `log_capital_gain = log1p(capital-gain)`.
`ColumnTransformer` sau đó: `StandardScaler` cho `age`/`hours-per-week`/`log_capital_gain`, drop
mọi cột phân loại thô, `passthrough` cho các cờ nhị phân + `education-num`. Toàn bộ gói trong 1
`Pipeline` — **chính pipeline này được WK3 và WK4 tái sử dụng NGUYÊN VẸN**, xem
[[sklearn-column-transformer-pipeline]].
<br><span class="en">`CustomFeatureCreator` (subclasses `BaseEstimator, TransformerMixin`)
creates 0/1 flags (`workclass_gov`, `is_married`, `race_white`...) plus
`log_capital_gain = log1p(capital-gain)`. `ColumnTransformer` then: `StandardScaler` on
`age`/`hours-per-week`/`log_capital_gain`, drops every raw categorical column, `passthrough`
for the binary flags + `education-num`. All wrapped in a single `Pipeline` —
**this exact pipeline is reused UNCHANGED by WK3 and WK4**, see
[[sklearn-column-transformer-pipeline]].</span>

### Modelling — baseline Logistic Regression & Random Forest

`log_reg_model = LogisticRegression(random_state=42, max_iter=1000)` và
`rf_model = RandomForestClassifier(random_state=42)`, cả hai fit trực tiếp trên
`Train_X_transformed`. Đánh giá bằng `classification_report` + confusion matrix +
`cross_val_score(cv=10, scoring='f1_weighted')`.
<br><span class="en">`log_reg_model = LogisticRegression(random_state=42, max_iter=1000)` and
`rf_model = RandomForestClassifier(random_state=42)`, both fit directly on
`Train_X_transformed`. Evaluated via `classification_report` + confusion matrix +
`cross_val_score(cv=10, scoring='f1_weighted')`.</span>

## Kết quả/số liệu nổi bật - <span class="en">Key Numbers</span>

| Chỉ số - <span class="en">Metric</span> | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy (test set) | 0.83 | 0.84 |
| F1 lớp `>50K` | 0.64 | 0.64 |
| 10-fold CV f1_weighted (trung bình) | 0.8298 | 0.8319 |
| Số đặc trưng sau feature engineering | 19 | 19 |
| Train/Test size (chỉ US) | 35,065 / 8,767 | — |

## Vấn đề/quirk phát hiện trong code gốc - <span class="en">Quirks / Issues Found</span>

Cả hai model đạt accuracy ~83-84% nhưng chỉ 0.64 F1 trên lớp thiểu số `>50K` — dấu hiệu rõ của
mất cân bằng lớp (~76/24) chưa được xử lý ở tuần này (accuracy "trông ổn" nhưng đánh lừa). Đây
chính là động lực trực tiếp khiến WK3 chuyển hẳn sang đo bằng **AUPRC** thay vì accuracy/F1
đơn thuần, xem [[wk3-notebook]].
<br><span class="en">Both models reach ~83-84% accuracy but only 0.64 F1 on the minority
`>50K` class — a clear sign of class imbalance (~76/24) not yet addressed this week (accuracy
"looks fine" but is misleading). This is the direct motivation for WK3 switching entirely to
**AUPRC** instead of plain accuracy/F1, see [[wk3-notebook]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-data-preparation-feature-engineering]] · [[wk2-practice]] · [[wk1-notebook]] ·
[[wk3-notebook]] · [[sklearn-column-transformer-pipeline]] ·
[[hierarchical-clustering-categorical-grouping]] · [[random-forest]] · [[overview]]
