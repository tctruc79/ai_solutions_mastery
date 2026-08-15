---
type: source
title: "WK3 Notebook — Modelling II (9 Models, XGBoost, Imbalance, Deep Learning)"
tags: [modelling, xgboost, imbalanced-data, deep-learning, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE - ADASync - CTGANs.ipynb"
---

## Thông tin nguồn - <span class="en">Source Info</span>

File `honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE -
ADASync - CTGANs.ipynb`, 64 cell. Thư viện MỚI so với tuần 2: `xgboost`, `tensorflow.keras`,
`ctgan.CTGAN`, `imblearn.over_sampling.{SMOTE, ADASYN}`, `imblearn.under_sampling.
RandomUnderSampler`, `hyperopt`, thêm nhiều `sklearn.ensemble` khác (GradientBoosting, AdaBoost,
ExtraTrees, Bagging), `sklearn.naive_bayes.GaussianNB`.
<br><span class="en">File `honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep
Learning - SMOTE - ADASync - CTGANs.ipynb`, 64 cells. Libraries NEW versus Week 2: `xgboost`,
`tensorflow.keras`, `ctgan.CTGAN`, `imblearn.over_sampling.{SMOTE, ADASYN}`,
`imblearn.under_sampling.RandomUnderSampler`, `hyperopt`, more `sklearn.ensemble` classes
(GradientBoosting, AdaBoost, ExtraTrees, Bagging), `sklearn.naive_bayes.GaussianNB`.</span>

## Mục tiêu buổi học - <span class="en">Learning Objectives</span>

Notebook chia 2 phần lớn: **Feature Engineering** (tái sử dụng NGUYÊN VẸN pipeline từ WK2) →
**Modelling Part I** (nhắc lại LR + RF từ WK2) → **Modelling Part II** (so sánh 9 model bằng
AUPRC, đào sâu XGBoost, xử lý mất cân bằng lớp, tuning Hyperopt, Deep Learning + CTGAN).
<br><span class="en">The notebook splits into 2 major parts: **Feature Engineering** (reuses
the WK2 pipeline UNCHANGED) → **Modelling Part I** (recaps LR + RF from WK2) → **Modelling Part
II** (compare 9 models via AUPRC, XGBoost deep-dive, class-imbalance handling, Hyperopt tuning,
Deep Learning + CTGAN).</span>

## Cấu trúc notebook - <span class="en">Notebook Structure</span>

- **Feature Engineering** — y hệt WK2 (`generate_train_test_data`, `CustomFeatureCreator`,
  `ColumnTransformer`, `Pipeline`).
  <br><span class="en">**Feature Engineering** — identical to WK2 (`generate_train_test_data`,
  `CustomFeatureCreator`, `ColumnTransformer`, `Pipeline`).</span>
- **Modelling Part I** — nhắc lại LogisticRegression + RandomForest (kết quả giống hệt WK2, vì
  cùng pipeline + cùng random_state).
  <br><span class="en">**Modelling Part I** — recaps LogisticRegression + RandomForest (results
  identical to WK2, same pipeline + same random_state).</span>
- **Modelling Part II** → "Comparison of 9 Different Models" (Naive Bayes, LR, Decision Tree,
  RF, Gradient Boosting, XGBoost, Extra Trees, AdaBoost, BaggingClassifier — đo bằng AUPRC qua
  10-fold `StratifiedKFold`) → "XGBoost" (huấn luyện + validate với custom AUPRC eval metric,
  xây model cuối, feature importance theo `gain`) → "XGBoost for Imbalanced Data" (so sánh
  SMOTE/ADASYN/Random Undersampling/dữ liệu gốc, rồi Hyperopt tuning trên chiến lược tốt nhất) →
  "Deep Learning with Tensorflow/Keras" (model cơ bản, rồi so sánh CTGAN oversampling vs
  undersampling vs không xử lý).
  <br><span class="en">**Modelling Part II** → "Comparison of 9 Different Models" (Naive Bayes,
  LR, Decision Tree, RF, Gradient Boosting, XGBoost, Extra Trees, AdaBoost, BaggingClassifier —
  measured by AUPRC via 10-fold `StratifiedKFold`) → "XGBoost" (train + validate with a custom
  AUPRC eval metric, build the final model, feature importance by `gain`) → "XGBoost for
  Imbalanced Data" (compare SMOTE/ADASYN/Random Undersampling/original data, then Hyperopt
  tuning on the best strategy) → "Deep Learning with Tensorflow/Keras" (base model, then compare
  CTGAN oversampling vs undersampling vs no treatment).</span>

## Thư viện & công cụ - <span class="en">Libraries & Tools</span>

`xgboost` (cả API `XGBClassifier` sklearn-style lẫn API gốc `xgb.train`/`xgb.DMatrix`),
`imbalanced-learn` (`SMOTE`, `ADASYN`, `RandomUnderSampler`), `hyperopt` (`fmin`, `tpe.suggest`,
`hp`, `Trials`), `tensorflow.keras` (`Sequential`, `Dense`, `Dropout`, `EarlyStopping`), `ctgan`
(`CTGAN` — Conditional Tabular GAN để sinh dữ liệu tổng hợp).
<br><span class="en">`xgboost` (both the sklearn-style `XGBClassifier` API and the native
`xgb.train`/`xgb.DMatrix` API), `imbalanced-learn` (`SMOTE`, `ADASYN`, `RandomUnderSampler`),
`hyperopt` (`fmin`, `tpe.suggest`, `hp`, `Trials`), `tensorflow.keras` (`Sequential`, `Dense`,
`Dropout`, `EarlyStopping`), `ctgan` (`CTGAN` — Conditional Tabular GAN for synthetic data
generation).</span>

## Walkthrough kỹ thuật theo từng phần - <span class="en">Technical Walkthrough</span>

### So sánh 9 model — chuyển hẳn sang đo bằng AUPRC

Thay vì accuracy/F1 như WK2, mọi so sánh model từ đây dùng
`make_scorer(average_precision_score, needs_proba=True)` qua `StratifiedKFold(n_splits=10)` —
xem [[auprc]]. Kết quả: XGBoost (0.7976) và Gradient Boosting (0.7962) dẫn đầu, Decision Tree
(0.5069) và Extra Trees (0.6079) yếu nhất trên dữ liệu GỐC (chưa xử lý mất cân bằng).
<br><span class="en">Instead of accuracy/F1 like WK2, every model comparison from here on uses
`make_scorer(average_precision_score, needs_proba=True)` via `StratifiedKFold(n_splits=10)` —
see [[auprc]]. Result: XGBoost (0.7976) and Gradient Boosting (0.7962) lead, Decision Tree
(0.5069) and Extra Trees (0.6079) are weakest on the ORIGINAL data (before imbalance
treatment).</span>

### XGBoost — API gốc với custom eval metric

Dùng `xgb.DMatrix` + `xgb.train()` (không phải `XGBClassifier.fit()`) để có thể truyền
`custom_metric=auprc_eval` (hàm tự viết tính AUPRC mỗi vòng boosting) và `scale_pos_weight` (tỷ
lệ lớp âm/dương) để mô hình tự cân bằng phần nào mà chưa cần resampling — xem [[xgboost]].
<br><span class="en">Uses `xgb.DMatrix` + `xgb.train()` (not `XGBClassifier.fit()`) to pass a
`custom_metric=auprc_eval` (a hand-written function computing AUPRC every boosting round) and
`scale_pos_weight` (negative/positive class ratio) so the model self-balances somewhat even
before any resampling — see [[xgboost]].</span>

### Xử lý mất cân bằng — SMOTE vs ADASYN vs Undersampling

`SMOTE(random_state=42).fit_resample(X, y)`, `ADASYN(...)`, `RandomUnderSampler(...)` — mỗi kỹ
thuật cho ra 1 tập train mới cân bằng hơn, đánh giá lại bằng `cross_val_score(scoring=
'average_precision')`. Kết quả: **SMOTE thắng rõ rệt** (AUPRC ≈ 0.959 so với dữ liệu gốc), vì
vậy `best_X_train, best_y_train = Train_X_smote, Train_y_smote` được chọn cho bước tuning tiếp
theo — xem [[imbalanced-data-resampling]].
<br><span class="en">`SMOTE(random_state=42).fit_resample(X, y)`, `ADASYN(...)`,
`RandomUnderSampler(...)` — each yields a new, more balanced training set, re-evaluated via
`cross_val_score(scoring='average_precision')`. Result: **SMOTE wins clearly** (AUPRC ≈ 0.959
versus original data), so `best_X_train, best_y_train = Train_X_smote, Train_y_smote` is chosen
for the next tuning step — see [[imbalanced-data-resampling]].</span>

### Hyperopt — tìm siêu tham số bằng TPE thay vì grid search

`fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50)` — thuật toán
Tree-structured Parzen Estimator (TPE) chọn bộ tham số tiếp theo THÔNG MINH dựa trên kết quả các
lần thử trước, thay vì duyệt lưới đều (grid search) hay ngẫu nhiên hoàn toàn — xem
[[hyperopt-tpe]]. Kết quả: AUPRC tăng từ 0.959 (SMOTE, tham số mặc định) lên 0.9635 (SMOTE +
tuned).
<br><span class="en">`fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50)` —
the Tree-structured Parzen Estimator (TPE) algorithm INTELLIGENTLY picks the next parameter set
based on previous trial results, instead of an even grid sweep (grid search) or pure randomness
— see [[hyperopt-tpe]]. Result: AUPRC rises from 0.959 (SMOTE, default params) to 0.9635 (SMOTE
+ tuned).</span>

### Deep Learning — mô hình cơ bản BỊ LỖI, rồi CTGAN oversampling

Mô hình Keras đầu tiên (`Dense(64)→Dropout→Dense(32)→Dropout→Dense(2, activation='tanh')` +
`categorical_crossentropy`) cho **loss = nan** ngay từ epoch 1 — lỗi kỹ thuật thực sự trong
notebook gốc (kết hợp sai `tanh` output với `categorical_crossentropy`). Phần sau dùng model
khác đúng hơn (`sigmoid` + `binary_crossentropy`) để so sánh Base Model vs **CTGAN Oversampling**
(sinh mẫu tổng hợp lớp thiểu số bằng GAN) vs Undersampling — xem [[ctgan]].
<br><span class="en">The first Keras model (`Dense(64)→Dropout→Dense(32)→Dropout→
Dense(2, activation='tanh')` + `categorical_crossentropy`) produces **loss = nan** from epoch 1
— a genuine technical bug in the original notebook (incorrectly pairing `tanh` output with
`categorical_crossentropy`). The following section uses a more correct model
(`sigmoid` + `binary_crossentropy`) to compare Base Model vs **CTGAN Oversampling** (GAN-generated
synthetic minority samples) vs Undersampling — see [[ctgan]].</span>

## Kết quả/số liệu nổi bật - <span class="en">Key Numbers</span>

| Model / kỹ thuật - <span class="en">Model / technique</span> | AUPRC |
|---|---|
| XGBoost (dữ liệu gốc, chưa xử lý mất cân bằng) | 0.7976 |
| Gradient Boosting (dữ liệu gốc) | 0.7962 |
| Decision Tree (dữ liệu gốc, yếu nhất) | 0.5069 |
| XGBoost + SMOTE | 0.9590 |
| XGBoost + ADASYN | ~0.949 |
| XGBoost + SMOTE + Hyperopt tuning | 0.9635 |
| Deep Learning cơ bản (lỗi `tanh`+`categorical_crossentropy`) | loss = NaN |
| Deep Learning + CTGAN oversampling (1 fold minh họa) | AUPRC ≈ 0.728 |

## Vấn đề/quirk phát hiện trong code gốc - <span class="en">Quirks / Issues Found</span>

Mô hình Deep Learning đầu tiên trong mục "Build Base Deep Learning Model" cho `loss: nan` xuyên
suốt cả 20 epoch — đây là lỗi kỹ thuật thật (không phải cố ý minh họa): output layer dùng
`activation='tanh'` (miền giá trị [-1,1]) nhưng hàm mất mát `categorical_crossentropy` yêu cầu
xác suất trong [0,1] — kết hợp sai gây NaN ngay từ vòng lặp đầu. Notebook không sửa lại cell này,
chỉ chuyển sang kiến trúc khác (`sigmoid`+`binary_crossentropy`) ở phần sau — bài học được đưa
vào [[wk3-modelling-imbalance-deep-learning]] như 1 pitfall thực hành.
<br><span class="en">The first Deep Learning model in "Build Base Deep Learning Model" produces
`loss: nan` throughout all 20 epochs — a genuine technical bug (not intentional
illustration): the output layer uses `activation='tanh'` (range [-1,1]) but the
`categorical_crossentropy` loss requires probabilities in [0,1] — this mismatch causes NaN from
the very first round. The notebook doesn't fix this cell, it simply switches to a different
architecture (`sigmoid`+`binary_crossentropy`) later — the lesson is carried into
[[wk3-modelling-imbalance-deep-learning]] as a practical pitfall.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-modelling-imbalance-deep-learning]] · [[wk2-notebook]] · [[wk4-notebook]] · [[xgboost]] ·
[[auprc]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] · [[overview]]
