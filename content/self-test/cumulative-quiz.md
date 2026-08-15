---
type: self-test
title: "Cumulative Self-Test — Tổng Hợp Xuyên Tuần (WK1–WK4)"
tags: [self-test, cumulative, wk1, wk2, wk3, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Tài liệu SỐNG, nay đã phủ ĐỦ 4 tuần (WK1–WK4) của môn học. Vai trò tương đương `exam-prep.md` bên
schema DBC, nhưng đây là bộ tra cứu nhanh + câu hỏi so sánh XUYÊN TUẦN, không thay thế bộ quiz
riêng từng tuần ([[wk1-quiz]], [[wk2-quiz]], [[wk3-quiz]], [[wk4-quiz]]). Mọi so sánh dưới đây cố
tình bắt đầu từ WK1 khi có thể — nhiều luận điểm tưởng như chỉ thuộc WK2–WK4 thực ra đã có gốc rễ
từ WK1 (dữ liệu thiếu ngụy trang → data leakage; fairness đặt câu hỏi sớm → FairLearn xử lý).
<br><span class="en">A LIVING document, now covering the FULL 4 weeks (WK1–WK4) of the course.
Plays the role DBC's `exam-prep.md` plays there, but this is a quick-reference + CROSS-WEEK
comparison question bank, not a replacement for each week's own quiz ([[wk1-quiz]], [[wk2-quiz]],
[[wk3-quiz]], [[wk4-quiz]]). Every comparison below deliberately starts from WK1 where possible —
many points that seem to belong only to WK2–WK4 actually have roots in WK1 (disguised missing
data → data leakage; fairness raised early → FairLearn addresses it).</span>

## Tra cứu nhanh - <span class="en">Quick Reference</span>

| Tuần - <span class="en">Week</span> | Trọng tâm - <span class="en">Focus</span> | Thước đo chính - <span class="en">Main metric</span> | Kỹ thuật mới - <span class="en">New technique</span> | Pitfall chính - <span class="en">Key pitfall</span> |
|---|---|---|---|---|
| WK1 | [[wk1-data-understanding]] — CRISP-DM: Collect→Describe→Explore→Verify | mô tả/khám phá, chưa modelling | [[point-biserial-correlation]], [[network-graph-categorical-eda]] | dữ liệu thiếu "ngụy trang" (vd `?` không phải NaN thật) |
| WK2 | [[wk2-data-preparation-feature-engineering]] — Data Prep 4 bước + baseline model | accuracy, F1 (per-class) | [[sklearn-column-transformer-pipeline]], [[hierarchical-clustering-categorical-grouping]] | accuracy cao (83-84%) là ảo giác khi mất cân bằng lớp |
| WK3 | [[wk3-modelling-imbalance-deep-learning]] — 9 model, XGBoost, resampling, Deep Learning | AUPRC (chính thức từ đây) | [[imbalanced-data-resampling]] (SMOTE/ADASYN/Undersampling), [[hyperopt-tpe]], [[ctgan]] | resample TRƯỚC rồi mới CV → rò rỉ dữ liệu tinh vi; activation lớp cuối không khớp loss → `nan` |
| WK4 | [[wk4-model-selection-explainability-fairness-deployment]] — lưới 54 tổ hợp, PCA/t-SNE/K-Means/DBSCAN, SHAP, FairLearn, Deployment | AUPRC trên CV **và** trên test set thật (2 con số, không phải 1) | [[pca-tsne-dimensionality-reduction]], [[kmeans-dbscan-clustering]], [[anomaly-detection]], [[shap]], [[fairlearn]] | CV cao (0.9639) trên dữ liệu resample ≠ test cao (0.7951) — bài học lớn nhất môn học |

## So sánh xuyên tuần - <span class="en">Cross-Week Comparisons</span>

### 1. Vì sao thước đo đổi nhiều lần: accuracy → F1 (per-class) → AUPRC (CV) → AUPRC (test set thật)

WK1 chưa modelling nên chưa cần thước đo hiệu năng — nhưng đã đặt nền bằng cách LÀM SẠCH target
đúng cách (`>50K.` → `>50K`), tiền đề để mọi thước đo sau này tính đúng. WK2 dùng accuracy trước,
phát hiện nó "trông tốt" (83-84%) nhưng F1 lớp thiểu số chỉ 0.64 → nhận ra accuracy gây hiểu lầm.
WK3 hành động cụ thể: chuyển hẳn sang AUPRC vì cả accuracy LẪN ROC-AUC đều có thể "trông tốt" ảo
trên dữ liệu mất cân bằng. WK4 đi xa hơn 1 bước nữa: ngay cả AUPRC cũng cần đo HAI LẦN — 1 lần
trên cross-validation (có thể lạc quan giả nếu dữ liệu đã resample), 1 lần trên test set THẬT
chưa từng resample — 2 con số này có thể lệch nhau rất xa (0.9639 vs 0.7951). Xem [[auprc]].
<br><span class="en">WK1 has no modelling yet — but lays the groundwork by correctly CLEANING the
target (`>50K.` → `>50K`), a prerequisite for every later metric to compute correctly. WK2 uses
accuracy first, discovers it "looks good" (83-84%) but minority-class F1 is only 0.64 → realises
accuracy is misleading. WK3 acts on it: switches fully to AUPRC since both accuracy AND ROC-AUC
can look deceptively good on imbalanced data. WK4 goes one step further: even AUPRC needs
measuring TWICE — once via cross-validation (can be falsely optimistic if data was resampled),
once on the REAL test set never resampled — these two numbers can diverge wildly (0.9639 vs
0.7951). See [[auprc]].</span>

### 2. Random Forest (WK2 baseline) vs XGBoost (WK3–WK4 model chính)

Cả 2 đều là ensemble cây, nhưng khác cơ chế: Random Forest = **bagging** (nhiều cây độc lập, huấn
luyện song song, biểu quyết đa số); XGBoost = **boosting** (cây tuần tự, mỗi cây sửa lỗi cây
trước). Trên cùng dữ liệu (bảng so sánh 9 model, WK3/WK4), XGBoost (0.7976 AUPRC) vượt rõ Random
Forest (0.7141 AUPRC) — minh chứng thực nghiệm cho việc boosting thường mạnh hơn bagging trên bài
toán khó/mất cân bằng, và là lý do XGBoost được chọn làm model trung tâm cho cả SHAP LẪN FairLearn
ở WK4. Xem [[xgboost]], [[random-forest]].
<br><span class="en">Both are tree ensembles but differ in mechanism: Random Forest = **bagging**
(many independent trees, trained in parallel, majority vote); XGBoost = **boosting** (sequential
trees, each correcting the previous one's errors). On the same data (WK3/WK4's 9-model
comparison), XGBoost (0.7976 AUPRC) clearly beats Random Forest (0.7141 AUPRC) — empirical
evidence that boosting typically outperforms bagging on hard/imbalanced problems, and the reason
XGBoost is chosen as the central model for both SHAP AND FairLearn in WK4. See [[xgboost]],
[[random-forest]].</span>

### 3. Xử lý mất cân bằng lớp — 4 tầng chi phí, và ai thắng thật sự (kết luận đầy đủ ở WK4)

(1) **`class_weight='balanced'`/`scale_pos_weight`** — không đổi dữ liệu, chỉ đổi trọng số học,
rẻ nhất. (2) **SMOTE/ADASYN/Undersampling** — đổi dữ liệu bằng phép tính hình học tức thời, chi
phí trung bình. (3) **PCA Features** — không phải kỹ thuật cân bằng lớp thực sự, nhưng WK4 đưa
vào cùng lưới so sánh như 1 biến thể biểu diễn dữ liệu. (4) **CTGAN** — huấn luyện 1 mạng GAN,
đắt nhất. WK3 kết luận SMOTE thắng trên CV (0.959); lưới 54 tổ hợp đầy đủ ở WK4 cho thấy CTGAN
(0.9639) NHỈNH HƠN SMOTE (0.959) trên CV — nhưng khi kiểm tra trên TEST SET THẬT, CTGAN
(0.7951) lại THUA cả model KHÔNG xử lý gì (0.8024)! Kết luận cuối cùng: trên CHÍNH bài toán này,
không có bằng chứng chắc chắn rằng BẤT KỲ kỹ thuật resampling nào cải thiện hiệu năng thật so với
việc chỉ dùng `scale_pos_weight` — bài học then chốt của cả môn học. Xem [[ctgan]],
[[imbalanced-data-resampling]].
<br><span class="en">(1) **`class_weight='balanced'`/`scale_pos_weight`** — doesn't alter data,
cheapest. (2) **SMOTE/ADASYN/Undersampling** — alters data via instant geometric computation,
medium cost. (3) **PCA Features** — not a true imbalance technique, but WK4 includes it in the
same comparison grid as a data-representation variant. (4) **CTGAN** — trains a GAN, most
expensive. WK3 concludes SMOTE wins on CV (0.959); the full 54-combination grid in WK4 shows
CTGAN (0.9639) edging out SMOTE (0.959) on CV — but when checked on the REAL TEST SET, CTGAN
(0.7951) actually LOSES to the model with NO treatment at all (0.8024)! Final conclusion: on THIS
specific problem, there's no solid evidence that ANY resampling technique improves real
performance over simply using `scale_pos_weight` — the pivotal lesson of the whole course. See
[[ctgan]], [[imbalanced-data-resampling]].</span>

### 4. Data leakage — 1 chủ đề xuất hiện lặp lại dưới nhiều hình thức, từ WK2 tới WK4

WK2: rò rỉ nếu `ColumnTransformer.fit()` chạy trên cả test set (thay vì chỉ train) — xem
[[wk2-data-preparation-feature-engineering]]. WK3: rò rỉ tinh vi hơn nếu SMOTE resample TRƯỚC khi
chia CV fold — 1 mẫu tổng hợp trong fold validation có thể bắt nguồn từ mẫu thật trong fold train
— xem [[wk3-modelling-imbalance-deep-learning]] mục "Lỗi thường gặp". WK4: dạng "rò rỉ" RỘNG HƠN
— toàn bộ QUY TRÌNH đánh giá (không chỉ 1 bước fit) có thể lạc quan giả nếu không có bước xác
nhận cuối cùng trên dữ liệu HOÀN TOÀN tách biệt — XGBoost+CTGAN thắng CV nhưng thua test set thật
là hệ quả trực tiếp của cùng nguyên tắc rò rỉ này ở quy mô lớn hơn. Cả 3 đều là biến thể của cùng
1 nguyên tắc: bất kỳ bước "học từ dữ liệu" nào (fit scaler, fit resampler, tối ưu hyperparameter)
chỉ được thực hiện trên phần dữ liệu THỰC SỰ dùng để huấn luyện, và phải luôn có 1 tập giữ lại
HOÀN TOÀN không chạm tới cho tới bước đánh giá cuối cùng.
<br><span class="en">WK2: leakage if `ColumnTransformer.fit()` runs on the combined/test set —
see [[wk2-data-preparation-feature-engineering]]. WK3: a subtler leakage if SMOTE resamples
BEFORE the CV fold split — see [[wk3-modelling-imbalance-deep-learning]], "Pitfalls" section.
WK4: a BROADER form of "leakage" — the entire evaluation WORKFLOW (not just one fit step) can be
falsely optimistic without a final confirmation step on data COMPLETELY held out — XGBoost+CTGAN
winning CV but losing on the real test set is a direct consequence of the same leakage principle
at a larger scale. All 3 are variants of one principle: any "learns from data" step (fitting a
scaler, fitting a resampler, tuning hyperparameters) must only run on data genuinely used for
training, and a completely untouched holdout set must always remain until the final evaluation
step.</span>

### 5. Fairness — câu hỏi đặt ra sớm nhất môn học (WK1), công cụ xử lý muộn nhất (WK4)

WK1 đã ghi nhận lệch nhân khẩu học đáng kể TRƯỚC KHI có bất kỳ model nào (Nam 66.85%/Nữ 33.15%,
White 85.50%) và đặt câu hỏi fairness ngay từ khâu khám phá dữ liệu — xem
[[wk1-data-understanding]] mục 6. Suốt WK2–WK3, câu hỏi này TẠM GÁC LẠI trong khi tập trung vào
độ chính xác dự đoán. WK4 mới quay lại xử lý bằng công cụ cụ thể — [[fairlearn]] đo `MetricFrame`
theo đúng các nhóm nhân khẩu học đã lệch từ WK1 (`is_male`, `race_white/black/...`), rồi giảm
thiểu bằng `ExponentiatedGradient`/`ThresholdOptimizer`/điều chỉnh thủ công. Bài học: đặt câu hỏi
fairness SỚM (WK1) không có nghĩa XỬ LÝ sớm — nhưng đặt sớm giúp biết CHÍNH XÁC cần đo nhóm nào
khi công cụ xử lý sẵn sàng.
<br><span class="en">WK1 already flagged meaningful demographic skew BEFORE any model existed
(Male 66.85%/Female 33.15%, White 85.50%) and raised the fairness question right at the
data-exploration stage — see [[wk1-data-understanding]] section 6. Throughout WK2–WK3, this
question was TEMPORARILY SET ASIDE while focus stayed on predictive accuracy. Only WK4 comes back
to address it with concrete tooling — [[fairlearn]] measures `MetricFrame` across exactly the
demographic groups already flagged as skewed in WK1 (`is_male`, `race_white/black/...`), then
mitigates via `ExponentiatedGradient`/`ThresholdOptimizer`/manual adjustment. Lesson: raising the
fairness question EARLY (WK1) doesn't mean ADDRESSING it early — but raising it early means
knowing EXACTLY which groups to measure once mitigation tooling is ready.</span>

### 6. EDA của WK1 gieo mầm cho Feature Engineering của WK2

`network-graph-categorical-eda` ở WK1 khám phá quan hệ giữa `education` và `occupation` bằng đồ
thị mạng — quan sát này không phải để "cho vui": WK2's `CustomFeatureCreator` sau đó tạo hàng
loạt cờ nhị phân từ `occupation` (`occupation_executuive`, `occupation_admin`...) và
`workclass`/`race` — những nhóm được chọn không ngẫu nhiên, phản ánh trực giác đã hình thành từ
bước khám phá dữ liệu tuần 1. Bài học: EDA (WK1) không phải bước "làm cho biết" tách rời khỏi
modelling — nó trực tiếp ĐỊNH HÌNH những đặc trưng được tạo ra sau này.
<br><span class="en">WK1's `network-graph-categorical-eda` explores the relationship between
`education` and `occupation` via a network graph — this observation isn't "just for show": WK2's
`CustomFeatureCreator` later creates a series of binary flags from `occupation`
(`occupation_executuive`, `occupation_admin`...) and `workclass`/`race` — these groupings aren't
random, they reflect intuition already formed during Week 1's exploration. Lesson: EDA (WK1)
isn't a "nice to know" step separate from modelling — it directly SHAPES the features created
later.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-quiz]] · [[wk2-quiz]] · [[wk3-quiz]] · [[wk4-quiz]] · [[wk1-data-understanding]] ·
[[wk2-data-preparation-feature-engineering]] · [[wk3-modelling-imbalance-deep-learning]] ·
[[wk4-model-selection-explainability-fairness-deployment]] · [[overview]]
