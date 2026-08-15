---
type: source
title: "WK4 Notebook — Model Selection, AI Explainability, AI Fairness, Deployment"
tags: [model-selection, pca, tsne, kmeans, dbscan, anomaly-detection, shap, fairlearn, deployment, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk4-model selection-ai explainability-Deployment-updatedv2.ipynb"
---

## Thông tin nguồn - <span class="en">Source Info</span>

File `honeyg_ai-practitioner-wk4-model selection-ai explainability-Deployment-updatedv2.ipynb`,
197 cell — notebook DÀI NHẤT trong 4 tuần. Thư viện MỚI so với tuần 3: `shap`, `streamlit`,
`joblib`, `fairlearn` (`MetricFrame`, `ExponentiatedGradient`, `DemographicParity`,
`EqualizedOdds`, `TruePositiveRateParity`, `ThresholdOptimizer`), thêm từ `sklearn`:
`sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans, DBSCAN}`,
`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`,
`sklearn.utils.class_weight.compute_sample_weight`, `sklearn.metrics.ConfusionMatrixDisplay`.
<br><span class="en">File `honeyg_ai-practitioner-wk4-model selection-ai explainability-
Deployment-updatedv2.ipynb`, 197 cells — the LONGEST notebook across the 4 weeks. Libraries NEW
versus Week 3: `shap`, `streamlit`, `joblib`, `fairlearn` (`MetricFrame`, `ExponentiatedGradient`,
`DemographicParity`, `EqualizedOdds`, `TruePositiveRateParity`, `ThresholdOptimizer`), plus from
`sklearn`: `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans,
DBSCAN}`, `sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`,
`sklearn.utils.class_weight.compute_sample_weight`, `sklearn.metrics.ConfusionMatrixDisplay`.</span>

## Mục tiêu buổi học - <span class="en">Learning Objectives</span>

Notebook chia 6 phần lớn, đi từ "chọn model tốt nhất" tới "đưa model ra thực tế": **Modelling Part
I–III** (nhắc lại LR/RF/9-model/XGBoost/imbalance/Deep Learning từ WK3, rồi mở rộng thành lưới
lớn hơn) → **Modelling Part IV** (học không giám sát: PCA, t-SNE, K-Means, DBSCAN, phát hiện dị
thường) → **AI Explainability** (SHAP) → **AI Fairness** (FairLearn) → **Deployment Pipeline**
(chạy lại toàn bộ từ đầu bằng hàm đóng gói, so sánh **54 tổ hợp** model × chiến lược xử lý mất
cân bằng, kiểm tra lại trên TEST SET thật, tinh chỉnh Hyperopt, lưu model bằng `joblib` cho 1
app Streamlit riêng).
<br><span class="en">The notebook splits into 6 major parts, moving from "pick the best model"
to "ship it": **Modelling Part I–III** (recaps LR/RF/9-model/XGBoost/imbalance/Deep Learning from
WK3, then expands into a larger grid) → **Modelling Part IV** (unsupervised learning: PCA, t-SNE,
K-Means, DBSCAN, anomaly detection) → **AI Explainability** (SHAP) → **AI Fairness** (FairLearn)
→ **Deployment Pipeline** (reruns everything from scratch via packaged functions, compares a
**54-combination grid** of models × imbalance strategies, re-checks on the real TEST SET, retunes
with Hyperopt, saves the model via `joblib` for a separate Streamlit app).</span>

## Cấu trúc notebook - <span class="en">Notebook Structure</span>

- **Feature Engineering** — y hệt WK2/WK3.
  <br><span class="en">**Feature Engineering** — identical to WK2/WK3.</span>
- **Modelling Part I** — nhắc lại LR + RF (kết quả giống WK2/WK3).
  <br><span class="en">**Modelling Part I** — recaps LR + RF (same results as WK2/WK3).</span>
- **Modelling Part II** → "Comparison of 9 Different Models" (2 phiên bản code: "Previous
  Code"/"Updated Code" — bản sau thêm XGBoost vào bảng) → "XGBoost" (train/validate + feature
  importance `gain`) → "XGBoost for Imbalanced Data" (SMOTE thắng ADASYN/Undersample, Hyperopt
  tuning) → "Deep Learning with Tensorflow/Keras" (model cơ bản đã SỬA lỗi `tanh` của WK3 — dùng
  `sigmoid`/`binary_crossentropy` ngay từ đầu — rồi so sánh Base vs CTGAN Oversampling vs
  Undersampling cho cả Deep Learning LẪN XGBoost).
  <br><span class="en">**Modelling Part II** → "Comparison of 9 Different Models" (2 code
  versions: "Previous Code"/"Updated Code" — the latter adds XGBoost to the table) → "XGBoost"
  (train/validate + `gain` feature importance) → "XGBoost for Imbalanced Data" (SMOTE beats
  ADASYN/Undersample, Hyperopt tuning) → "Deep Learning with Tensorflow/Keras" (base model already
  FIXES WK3's `tanh` bug — uses `sigmoid`/`binary_crossentropy` from the start — then compares
  Base vs CTGAN Oversampling vs Undersampling for BOTH Deep Learning AND XGBoost).</span>
- **Modelling Part III** → "Finding Best Model: 45 Supervised Model + Imbalanced Data Solutions"
  — một bản nháp SỚM HƠN của lưới so sánh (chỉ 4 chiến lược: SMOTE/ADASYN/UNDERSAMPLE/ORIGINAL,
  9 model = 36 tổ hợp thực tế dù heading ghi "45") — bị THAY THẾ bởi lưới đầy đủ hơn ở phần
  Deployment Pipeline bên dưới.
  <br><span class="en">**Modelling Part III** → "Finding Best Model: 45 Supervised Model +
  Imbalanced Data Solutions" — an EARLIER draft of the comparison grid (only 4 strategies:
  SMOTE/ADASYN/UNDERSAMPLE/ORIGINAL, 9 models = 36 actual combinations despite the heading saying
  "45") — SUPERSEDED by the fuller grid in the Deployment Pipeline section below.</span>
- **Modelling Part IV** → "Dimensionality Reduction with PCA" (95% variance → 10 components) →
  "Dimensionality Reduction/Visualisation with TSNE" (có/không tiền xử lý PCA) → "K-Means
  Clustering" (elbow method) → "Clustering with DBSCAN" → "Unsupervised Anomaly Detection"
  (Isolation Forest, LOF, Autoencoder).
  <br><span class="en">**Modelling Part IV** → "Dimensionality Reduction with PCA" (95% variance
  → 10 components) → "Dimensionality Reduction/Visualisation with TSNE" (with/without PCA
  pre-processing) → "K-Means Clustering" (elbow method) → "Clustering with DBSCAN" →
  "Unsupervised Anomaly Detection" (Isolation Forest, LOF, Autoencoder).</span>
- **AI Explainability** → "Global Explainability" (SHAP `summary_plot` beeswarm + bar) → "Local
  Explainability" (SHAP `waterfall` cho từng dòng dữ liệu cụ thể).
  <br><span class="en">**AI Explainability** → "Global Explainability" (SHAP `summary_plot`
  beeswarm + bar) → "Local Explainability" (SHAP `waterfall` for individual rows).</span>
- **AI Fairness** → "FairLearn" (`MetricFrame` đo theo nhóm, `ExponentiatedGradient` với 3 ràng
  buộc, `ThresholdOptimizer`, điều chỉnh ngưỡng thủ công theo nhóm).
  <br><span class="en">**AI Fairness** → "FairLearn" (`MetricFrame` measured by group,
  `ExponentiatedGradient` with 3 constraints, `ThresholdOptimizer`, manual per-group threshold
  adjustment).</span>
- **Deployment Pipeline** → chạy lại TOÀN BỘ pipeline bằng các hàm đóng gói (`generate_train_test_
  data`, `build_features`...) → "Modeling, Model Selection and Model Fine-Tunning" (lưới **54 tổ
  hợp** = 9 model × 6 chiến lược: SMOTE/ADASYN/UNDERSAMPLE/PCA Features/CTGANS
  Oversampling/ORIGINAL) → "Selecting the Best Model - Testing on Test Set" (kiểm tra lại trên
  TEST SET thật) → "Hyperparameter Tunning for Best Model" (Hyperopt lần 2) → "Saving the Best
  Model" (`joblib.dump`) → rebuild SHAP + FairLearn trên model cuối → lưu `Test_X_transformed`,
  `Test_y_transformed`, `feature_names` ra file `.npy` cho 1 app Streamlit riêng.
  <br><span class="en">**Deployment Pipeline** → reruns the ENTIRE pipeline via packaged functions
  (`generate_train_test_data`, `build_features`...) → "Modeling, Model Selection and Model
  Fine-Tunning" (**54-combination grid** = 9 models × 6 strategies: SMOTE/ADASYN/UNDERSAMPLE/PCA
  Features/CTGANS Oversampling/ORIGINAL) → "Selecting the Best Model - Testing on Test Set"
  (re-checks on the real TEST SET) → "Hyperparameter Tunning for Best Model" (second Hyperopt
  pass) → "Saving the Best Model" (`joblib.dump`) → rebuilds SHAP + FairLearn on the final model
  → saves `Test_X_transformed`, `Test_y_transformed`, `feature_names` to `.npy` files for a
  separate Streamlit app.</span>

## Thư viện & công cụ - <span class="en">Libraries & Tools</span>

`shap` (`TreeExplainer`, `summary_plot`, `plots.waterfall`), `fairlearn.metrics.MetricFrame`,
`fairlearn.reductions.{ExponentiatedGradient, DemographicParity, EqualizedOdds,
TruePositiveRateParity}`, `fairlearn.postprocessing.ThresholdOptimizer`,
`sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans, DBSCAN}`,
`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `joblib`, `streamlit`.
<br><span class="en">`shap` (`TreeExplainer`, `summary_plot`, `plots.waterfall`),
`fairlearn.metrics.MetricFrame`, `fairlearn.reductions.{ExponentiatedGradient, DemographicParity,
EqualizedOdds, TruePositiveRateParity}`, `fairlearn.postprocessing.ThresholdOptimizer`,
`sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.cluster.{KMeans, DBSCAN}`,
`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `joblib`,
`streamlit`.</span>

## Walkthrough kỹ thuật theo từng phần - <span class="en">Technical Walkthrough</span>

### Lưới 54 tổ hợp — 9 model × 6 chiến lược xử lý mất cân bằng

Phần "Deployment Pipeline" chạy lại toàn bộ so sánh với 6 chiến lược: SMOTE, ADASYN,
Undersampling, **PCA Features** (9 thành phần), **CTGAN Oversampling**, và dữ liệu gốc
(ORIGINAL) — cho MỖI 1 trong 9 model → 54 tổ hợp. Kết quả (bảng `model_list`, sắp theo AUPRC
giảm dần): **XGBoost + CTGAN (0.9639)** dẫn đầu, sát nút Gradient Boosting + CTGAN (0.9634),
rồi XGBoost + SMOTE (0.9590), XGBoost + ADASYN (0.9507).
<br><span class="en">The "Deployment Pipeline" section reruns the full comparison with 6
strategies: SMOTE, ADASYN, Undersampling, **PCA Features** (9 components), **CTGAN
Oversampling**, and the original data (ORIGINAL) — for EACH of 9 models → 54 combinations.
Result (the `model_list` table, sorted by descending AUPRC): **XGBoost + CTGAN (0.9639)** leads,
closely followed by Gradient Boosting + CTGAN (0.9634), then XGBoost + SMOTE (0.9590), XGBoost +
ADASYN (0.9507).</span>

### Kiểm tra trên TEST SET thật — điểm ngoặt của cả môn học

Mục "Selecting the Best Model - Testing on Test Set" huấn luyện XGBoost trên dữ liệu GỐC (không
resample) rồi dự đoán trên `Test_X_transformed` (tập test THẬT, chưa từng resample) → **AUC-PR =
0.8024**. Lặp lại với XGBoost huấn luyện trên dữ liệu ĐÃ CTGAN-resample → dự đoán trên CÙNG tập
test thật đó → **AUC-PR = 0.7951** — THẤP HƠN bản không resample, dù CTGAN vừa "thắng" với AUPRC
0.9639 trên cross-validation! Đây chính là bài học được [[auprc]] và [[xgboost]] cảnh báo trước từ
[[wk3-modelling-imbalance-deep-learning]]: AUPRC đo trên dữ liệu ĐÃ resample (dù qua CV) có thể
lạc quan giả, vì synthetic sample có thể "quá giống" phần dữ liệu gốc dùng để sinh ra nó — phải
luôn xác nhận lại bằng tập test THẬT, chưa từng resample.
<br><span class="en">The "Selecting the Best Model - Testing on Test Set" section trains XGBoost
on the ORIGINAL data (no resampling) then predicts on `Test_X_transformed` (the REAL test set,
never resampled) → **AUC-PR = 0.8024**. Repeated with XGBoost trained on CTGAN-resampled data →
predicting on that SAME real test set → **AUC-PR = 0.7951** — LOWER than the non-resampled
version, despite CTGAN having just "won" with a 0.9639 cross-validation AUPRC! This is exactly
the lesson foreshadowed by [[auprc]] and [[xgboost]] from [[wk3-modelling-imbalance-deep-
learning]]: AUPRC measured on resampled data (even via CV) can be falsely optimistic, since a
synthetic sample may be "too similar" to the real point it was generated from — always confirm
against a REAL test set that was never resampled.</span>

### PCA, t-SNE, K-Means, DBSCAN, phát hiện dị thường

PCA: 95% phương sai cần 10/19 thành phần (nhưng lưới 54 tổ hợp lại dùng 9 thành phần — không
khớp hoàn toàn với con số 10 vừa tìm được). t-SNE: giảm chiều phi tuyến để trực quan hóa 2D, thử
cả trên dữ liệu gốc lẫn sau PCA. K-Means: dùng elbow method rồi chọn `optimal_k = 10` — xem
[[kmeans-dbscan-clustering]]. DBSCAN: `eps=0.5, min_samples=40`, loại điểm nhiễu (label -1) trước
khi tính trung bình theo cụm. Anomaly Detection: Isolation Forest (AUPRC-kiểu 0.6161), LOF
(0.6082), Autoencoder tự viết (0.5753) — xem [[anomaly-detection]].
<br><span class="en">PCA: 95% variance needs 10/19 components (yet the 54-combination grid uses
9 components — not fully matching the 10 just found). t-SNE: non-linear dimensionality reduction
for 2D visualisation, tried on both the raw data and after PCA. K-Means: uses the elbow method
then picks `optimal_k = 10` — see [[kmeans-dbscan-clustering]]. DBSCAN: `eps=0.5, min_samples=40`,
drops noise points (label -1) before computing per-cluster means. Anomaly Detection: Isolation
Forest (AUPRC-like 0.6161), LOF (0.6082), a hand-written Autoencoder (0.5753) — see
[[anomaly-detection]].</span>

### SHAP — giải thích global và local

`shap.TreeExplainer(final_model)` trên model XGBoost cuối cùng → `shap.summary_plot` (beeswarm:
mọi điểm dữ liệu, mọi đặc trưng, cùng lúc — global) và `shap.plots.waterfall` (1 điểm dữ liệu cụ
thể, từng đặc trưng đóng góp bao nhiêu vào dự đoán đó — local) — xem [[shap]].
<br><span class="en">`shap.TreeExplainer(final_model)` on the final XGBoost model →
`shap.summary_plot` (beeswarm: every data point, every feature, simultaneously — global) and
`shap.plots.waterfall` (one specific data point, how much each feature contributed to that
prediction — local) — see [[shap]].</span>

### FairLearn — đo và giảm thiểu thiên lệch theo nhóm nhạy cảm

`MetricFrame` đo accuracy/precision/AUPRC/TPR TÁCH RIÊNG theo từng nhóm (`is_male`, `is_married`,
`race_white/black/asian-pac-islander`). Ba cách giảm thiểu: `ExponentiatedGradient` (in-processing
— ràng buộc ngay trong lúc huấn luyện, thử cả 3 constraint `DemographicParity`/`EqualizedOdds`/
`TruePositiveRateParity`), `ThresholdOptimizer` (post-processing — chỉnh ngưỡng phân loại sau khi
đã có model, theo `demographic_parity`), và điều chỉnh ngưỡng THỦ CÔNG theo nhóm (0.3 cho nhóm
chưa kết hôn, 0.7 cho nhóm đã kết hôn) — xem [[fairlearn]].
<br><span class="en">`MetricFrame` measures accuracy/precision/AUPRC/TPR SEPARATELY per group
(`is_male`, `is_married`, `race_white/black/asian-pac-islander`). Three mitigation approaches:
`ExponentiatedGradient` (in-processing — constrains during training itself, tried with all 3
constraints `DemographicParity`/`EqualizedOdds`/`TruePositiveRateParity`), `ThresholdOptimizer`
(post-processing — adjusts the classification threshold after the model already exists, per
`demographic_parity`), and MANUAL per-group threshold adjustment (0.3 for the not-married group,
0.7 for the married group) — see [[fairlearn]].</span>

## Kết quả/số liệu nổi bật - <span class="en">Key Numbers</span>

| Model / kỹ thuật - <span class="en">Model / technique</span> | AUPRC (CV) | AUPRC (test set thật) |
|---|---|---|
| XGBoost + CTGAN Oversampling (thắng lưới 54 tổ hợp) | 0.9639 | 0.7951 |
| Gradient Boosting + CTGAN Oversampling | 0.9634 | — |
| XGBoost + SMOTE | 0.9590 | — |
| XGBoost (dữ liệu gốc, không resample) | 0.7976 | **0.8024** |
| XGBoost + Hyperopt tuning (trên CTGAN, lần 2) | 0.9653 | chưa kiểm tra lại trong notebook |
| Isolation Forest (anomaly detection) | AUPRC-kiểu 0.6161 | — |
| LOF (anomaly detection) | AUPRC-kiểu 0.6082 | — |
| Autoencoder tự viết (anomaly detection) | AUPRC-kiểu 0.5753 | — |
| Optimal PCA components (95% variance) | 10/19 | (lưới 54 tổ hợp lại dùng 9) |
| Optimal K-Means k (chọn thủ công) | 10 | (comment code lại ghi "3") |

## Vấn đề/quirk phát hiện trong code gốc - <span class="en">Quirks / Issues Found</span>

Ba vấn đề kỹ thuật thật, không phải cố ý minh họa:
<br><span class="en">Three genuine technical issues, not intentional illustration:</span>

1. **Bug chọn "best model" bằng cách so sánh KÝ TỰ, không phải giá trị AUPRC** — dòng `best_model
   = max(results, key=lambda x: x[1])` lặp qua CÁC KHÓA (chuỗi tên model, vd `"XGBoost SMOTE"`)
   của dict `results` chứ không phải cặp (khóa, giá trị); `x[1]` do đó là KÝ TỰ THỨ 2 của tên
   model (vd `'G'` trong `"XGBoost..."`), không phải điểm AUPRC — kết quả in ra sai hoàn toàn:
   **"Best model based on AUPRC: Extra Trees SMOTE with an AUPRC of 0.9026"**, trong khi bảng
   `model_list` (sắp xếp đúng bằng `.sort_values()`) ngay bên dưới cho thấy người thắng THẬT là
   **XGBoost + CTGAN Oversampling (0.9639)**. Cách sửa đúng:
   `max(results.items(), key=lambda kv: kv[1][0])`.
2. **Model được lưu/triển khai chưa từng được xác nhận lại trên test set SAU KHI tinh chỉnh** —
   mục "Testing on Test Set" chỉ kiểm tra 2 model CHƯA tuning (gốc: 0.8024, CTGAN: 0.7951); sau
   đó Hyperopt tinh chỉnh tiếp trên dữ liệu CTGAN (đạt CV AUPRC 0.9653) và model này được
   `joblib.dump` làm bản triển khai cuối cùng — nhưng notebook KHÔNG quay lại đo AUPRC của
   chính bản đã tinh chỉnh này trên test set thật. Vì mục ngay phía trên đã cho thấy CV cao
   không đảm bảo test cao, đây là 1 khoảng trống đánh giá đáng chú ý.
3. **Elbow method cho K-Means: code không khớp comment** — comment trong cell viết "Choose k
   from the elbow plot visually, let's say it appears to be 3" nhưng dòng code ngay sau lại gán
   `optimal_k = 10` — không rõ đây là lỗi gõ nhầm hay thay đổi quyết định mà quên sửa comment,
   nhưng cho thấy 1 pitfall thực hành: luôn đối chiếu comment với code thực thi, đừng tin comment
   mù quáng.

<br>
<span class="en">1. **The "best model" is picked by comparing CHARACTERS, not AUPRC values** —
the line `best_model = max(results, key=lambda x: x[1])` iterates over the dict `results`' KEYS
(model-name strings, e.g. `"XGBoost SMOTE"`) rather than (key, value) pairs; `x[1]` is therefore
the 2nd CHARACTER of the model name (e.g. `'G'` in `"XGBoost..."`), not an AUPRC score — the
printed result is entirely wrong: **"Best model based on AUPRC: Extra Trees SMOTE with an AUPRC
of 0.9026"**, while the `model_list` table (correctly sorted via `.sort_values()`) right below it
shows the REAL winner is **XGBoost + CTGAN Oversampling (0.9639)**. Correct fix:
`max(results.items(), key=lambda kv: kv[1][0])`.
2. **The saved/deployed model is never re-checked on the test set AFTER tuning** — the "Testing
on Test Set" section only checks 2 UNTUNED models (original: 0.8024, CTGAN: 0.7951); Hyperopt
then further tunes on the CTGAN data (reaching 0.9653 CV AUPRC), and this model is `joblib.dump`ed
as the final deployed version — but the notebook never goes back to measure THIS tuned model's
AUPRC on the real test set. Since the section right above just demonstrated that high CV doesn't
guarantee high test performance, this is a notable evaluation gap.
3. **K-Means elbow method: code doesn't match the comment** — a cell comment reads "Choose k from
the elbow plot visually, let's say it appears to be 3" but the very next line sets `optimal_k =
10` — unclear whether this is a typo or a changed decision with a stale comment, but it
illustrates a practical pitfall: always cross-check comments against the code that actually runs,
never trust a comment blindly.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-model-selection-explainability-fairness-deployment]] · [[wk3-notebook]] · [[xgboost]] ·
[[auprc]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[pca-tsne-dimensionality-reduction]] · [[kmeans-dbscan-clustering]] · [[anomaly-detection]] ·
[[shap]] · [[fairlearn]] · [[overview]]
