---
type: concept
title: "FairLearn — Đo Lường & Giảm Thiểu Thiên Lệch Thuật Toán"
tags: [fairlearn, fairness, bias-mitigation, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**FairLearn** là thư viện Python đo lường và giảm thiểu **thiên lệch thuật toán** (algorithmic
bias) — kiểm tra xem model có đối xử KHÁC NHAU đáng kể giữa các nhóm nhạy cảm (`sensitive
features`, vd giới tính, chủng tộc, tình trạng hôn nhân) hay không, và cung cấp công cụ can thiệp
nếu có.
<br><span class="en">**FairLearn** is a Python library for measuring and mitigating
**algorithmic bias** — checking whether a model treats sensitive groups (`sensitive features`,
e.g. gender, race, marital status) MEANINGFULLY DIFFERENTLY, and providing intervention tools if
so.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Ngay từ [[wk1-data-understanding]], dữ liệu đã cho thấy lệch nhân khẩu học đáng kể (Nam 66.85%,
White 85.50%). Một model đạt AUPRC cao TRÊN TOÀN BỘ dữ liệu vẫn có thể ẩn giấu chênh lệch hiệu
năng LỚN giữa các nhóm — vd dự đoán chính xác cho nam giới nhưng kém hơn hẳn cho nữ giới. Nếu
model này dùng để ra quyết định thực (vay vốn, tuyển dụng), chênh lệch đó có thể VI PHẠM PHÁP
LUẬT hoặc gây tổn hại thực tế cho 1 nhóm người — FairLearn giúp PHÁT HIỆN chênh lệch này TRƯỚC
khi triển khai, và cung cấp cách GIẢM THIỂU nếu cần.
<br><span class="en">Since [[wk1-data-understanding]], the data has already shown meaningful
demographic skew (Male 66.85%, White 85.50%). A model achieving high AUPRC ON THE WHOLE DATASET
can still hide a LARGE performance gap between groups — e.g. predicting accurately for men but
much worse for women. If this model drives real decisions (loans, hiring), that gap could VIOLATE
THE LAW or cause real harm to one group — FairLearn helps DETECT this gap BEFORE deployment, and
provides ways to MITIGATE it if needed.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

**Đo lường**: `MetricFrame(metrics=..., y_true=..., y_pred=..., sensitive_features=...)` tính bất
kỳ chỉ số nào (accuracy, precision, AUPRC, TPR...) TÁCH RIÊNG theo từng giá trị của
`sensitive_features`, cho `.by_group` để so sánh trực tiếp. **Giảm thiểu — 3 điểm can thiệp khác
nhau trong pipeline**: (1) **In-processing** — `ExponentiatedGradient(estimator=..., constraints=
...)` huấn luyện LẠI model với ràng buộc công bằng đưa thẳng vào hàm mất mát; các ràng buộc gồm
`DemographicParity` (tỷ lệ dự đoán dương bằng nhau giữa nhóm — không quan tâm nhãn thật), `Equali
zedOdds` (TPR VÀ FPR bằng nhau giữa nhóm — quan tâm cả nhãn thật), `TruePositiveRateParity` (chỉ
TPR bằng nhau). (2) **Post-processing** — `ThresholdOptimizer(estimator=..., constraints=
"demographic_parity")` KHÔNG huấn luyện lại, chỉ tìm ngưỡng phân loại RIÊNG cho mỗi nhóm sao cho
đạt ràng buộc công bằng. (3) **Thủ công** — tự chọn ngưỡng khác nhau theo nhóm dựa trên quan sát
TPR/TNR ban đầu (vd notebook chọn 0.3 cho nhóm chưa kết hôn, 0.7 cho nhóm đã kết hôn) — đơn giản
nhất nhưng dễ chủ quan, không có đảm bảo toán học như 2 cách trên.
<br><span class="en">**Measurement**: `MetricFrame(metrics=..., y_true=..., y_pred=...,
sensitive_features=...)` computes any metric (accuracy, precision, AUPRC, TPR...) SEPARATELY per
value of `sensitive_features`, exposing `.by_group` for direct comparison. **Mitigation — 3
different intervention points in the pipeline**: (1) **In-processing** —
`ExponentiatedGradient(estimator=..., constraints=...)` RETRAINS the model with a fairness
constraint built directly into the loss; constraints include `DemographicParity` (equal
positive-prediction rate across groups — ignores true labels), `EqualizedOdds` (both TPR AND FPR
equal across groups — considers true labels), `TruePositiveRateParity` (only TPR equal). (2)
**Post-processing** — `ThresholdOptimizer(estimator=..., constraints="demographic_parity")` does
NOT retrain, just finds a SEPARATE classification threshold per group that satisfies the fairness
constraint. (3) **Manual** — choosing different per-group thresholds by hand based on observed
initial TPR/TNR (e.g. the notebook picks 0.3 for the not-married group, 0.7 for the married group)
— simplest but prone to subjectivity, with none of the mathematical guarantees of the two
approaches above.</span>

## Ví dụ - <span class="en">Example</span>

```python
from fairlearn.metrics import MetricFrame, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

# 1. Đo lường trước — phát hiện chênh lệch
mf = MetricFrame(metrics={'tpr': true_positive_rate, 'accuracy': accuracy_score},
                  y_true=y_true, y_pred=model.predict(X),
                  sensitive_features=X['binary__is_married'])
print(mf.by_group)     # TPR/accuracy riêng cho mỗi nhóm — so sánh trực tiếp

# 2. Giảm thiểu — in-processing (huấn luyện lại với ràng buộc)
mitigator = ExponentiatedGradient(estimator=xgb.XGBClassifier(), constraints=DemographicParity())
mitigator.fit(X, y_true, sensitive_features=X['binary__is_married'])

# 3. Giảm thiểu — post-processing (không huấn luyện lại)
optimizer = ThresholdOptimizer(estimator=model, constraints="demographic_parity",
                                predict_method='predict_proba')
optimizer.fit(X, y_true, sensitive_features=X['binary__is_married'])
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Dùng trong [[wk4-notebook]]/"AI Fairness" — đo theo 6 đặc trưng nhạy cảm (`is_male`, `is_married`,
`race_white/black/asian-pac-islander`, `age`), rồi thử cả 3 cách giảm thiểu trên đặc trưng
`is_married`. Kết nối trực tiếp tới lệch nhân khẩu học đã phát hiện từ [[wk1-data-understanding]]
— đây là nơi câu hỏi fairness đặt ra "sớm" ở tuần 1 CUỐI CÙNG được xử lý bằng công cụ cụ thể.
<br><span class="en">Used in [[wk4-notebook]]/"AI Fairness" — measured across 6 sensitive
features (`is_male`, `is_married`, `race_white/black/asian-pac-islander`, `age`), then all 3
mitigation approaches tried on the `is_married` feature. Directly connects to the demographic
skew already flagged in [[wk1-data-understanding]] — this is where the fairness question raised
"early" in Week 1 FINALLY gets addressed with concrete tooling.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

3 constraint chính không có "đáp án đúng duy nhất" — `DemographicParity` phù hợp khi mục tiêu là
CƠ HỘI ngang nhau bất kể nhãn thật (vd quota tuyển dụng); `EqualizedOdds`/`TruePositiveRateParity`
phù hợp khi mục tiêu là ĐỘ CHÍNH XÁC ngang nhau giữa nhóm (không "trừng phạt" nhóm có tỷ lệ nhãn
dương thực tế khác nhau) — chọn constraint nào phụ thuộc bối cảnh pháp lý/đạo đức cụ thể, không
phải quyết định thuần kỹ thuật. Liên quan [[shap]] — SHAP giúp ĐIỀU TRA nguyên nhân gây thiên
lệch (đặc trưng nào đóng góp bất thường cho 1 nhóm), còn FairLearn ĐO LƯỜNG kết quả cuối và
GIẢM THIỂU — 2 công cụ bổ sung nhau trong quy trình đảm bảo AI có trách nhiệm.
<br><span class="en">The 3 main constraints have no "single correct answer" —
`DemographicParity` suits contexts where the goal is equal OPPORTUNITY regardless of true label
(e.g. hiring quotas); `EqualizedOdds`/`TruePositiveRateParity` suit contexts where the goal is
equal ACCURACY across groups (not "penalising" a group with a genuinely different true positive
rate) — which constraint to pick depends on the specific legal/ethical context, not a purely
technical decision. Related to [[shap]] — SHAP helps INVESTIGATE the cause of bias (which feature
contributes abnormally for one group), while FairLearn MEASURES the end result and MITIGATES —
two complementary tools in a responsible-AI process.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[wk1-data-understanding]] · [[shap]] · [[xgboost]]
