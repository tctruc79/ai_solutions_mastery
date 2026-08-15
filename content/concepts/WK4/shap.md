---
type: concept
title: "SHAP (SHapley Additive exPlanations)"
tags: [shap, explainability, interpretability, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**SHAP** (SHapley Additive exPlanations) là phương pháp giải thích model dựa trên lý thuyết game
**Shapley value** — tính đóng góp CÔNG BẰNG của TỪNG đặc trưng vào TỪNG dự đoán cụ thể của bất kỳ
model nào (kể cả model "hộp đen" như XGBoost).
<br><span class="en">**SHAP** (SHapley Additive exPlanations) is a model-explanation method based
on **Shapley value** game theory — computing the FAIR contribution of EACH feature to EACH
specific prediction of any model (including "black-box" models like XGBoost).</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

`feature_importances_` (Random Forest) hay `gain` (XGBoost, xem [[xgboost]]) chỉ cho biết đặc
trưng nào QUAN TRỌNG NÓI CHUNG, tính trung bình trên toàn bộ dữ liệu — không trả lời được "tại
sao model dự đoán CHÍNH XÁC người NÀY có thu nhập >50K?". Khi cần GIẢI TRÌNH 1 quyết định cụ thể
(vd từ chối khoản vay, phải giải thích cho khách hàng) hoặc DEBUG model (phát hiện model dựa vào
đặc trưng không hợp lý cho 1 nhóm ca cụ thể), cần đi sâu xuống mức TỪNG DỰ ĐOÁN — đây là việc
SHAP làm được mà feature importance thông thường không làm được.
<br><span class="en">`feature_importances_` (Random Forest) or `gain` (XGBoost, see [[xgboost]])
only tell you which features matter OVERALL, averaged across the whole dataset — they don't
answer "why did the model predict THIS SPECIFIC person has income >50K?". When a SPECIFIC
decision needs to be justified (e.g. a loan rejection explained to a customer) or a model needs
DEBUGGING (catching it relying on an unreasonable feature for a specific group of cases), you need
to drill down to the level of INDIVIDUAL PREDICTIONS — this is what SHAP does that ordinary
feature importance cannot.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

Với 1 dự đoán cụ thể, SHAP coi mỗi đặc trưng như 1 "người chơi" trong 1 trò chơi hợp tác, và tính
đóng góp của đặc trưng đó = trung bình có trọng số của mức thay đổi dự đoán khi thêm đặc trưng đó
vào MỌI tổ hợp con có thể của các đặc trưng còn lại (Shapley value từ lý thuyết game) — đảm bảo
tính chất **additive**: tổng SHAP value của MỌI đặc trưng + giá trị nền (baseline, trung bình dự
đoán trên toàn dữ liệu) = ĐÚNG BẰNG dự đoán thực tế cho điểm đó. `shap.TreeExplainer` tính nhanh
chính xác cho model dạng cây (XGBoost, Random Forest) bằng thuật toán tối ưu riêng, không cần xấp
xỉ như các model khác.
<br><span class="en">For one specific prediction, SHAP treats each feature as a "player" in a
cooperative game, and computes that feature's contribution = a weighted average of how much the
prediction changes when adding that feature across EVERY possible subset of the remaining
features (Shapley value from game theory) — guaranteeing the **additive** property: the sum of
EVERY feature's SHAP value + a baseline value (the average prediction across all data) = EXACTLY
EQUALS the actual prediction for that point. `shap.TreeExplainer` computes this quickly and
exactly for tree-based models (XGBoost, Random Forest) via a specialised algorithm, no
approximation needed unlike other model types.</span>

## Ví dụ - <span class="en">Example</span>

```python
import shap

explainer = shap.TreeExplainer(final_xgb_model)

# Global — đặc trưng nào quan trọng nhất trên TOÀN BỘ dữ liệu
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)          # beeswarm
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names)

# Local — vì sao model dự đoán ĐÚNG DÒNG này như vậy
single_shap = explainer(X_sample.iloc[[3]])
shap.plots.waterfall(single_shap[0], max_display=19)
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Dùng trong [[wk4-notebook]]/"AI Explainability" trên model XGBoost cuối cùng (huấn luyện trên dữ
liệu SMOTE-resampled) — mục "Global Explainability" (`summary_plot`, lấy mẫu 200–500 dòng vì
tính SHAP cho toàn bộ dữ liệu rất tốn thời gian — notebook ghi chú rõ "up to 30 minutes") và
"Local Explainability" (`waterfall` cho từng dòng mẫu). SHAP explainer được `joblib.dump` cùng
model để dùng lại trong app Streamlit triển khai.
<br><span class="en">Used in [[wk4-notebook]]/"AI Explainability" on the final XGBoost model
(trained on SMOTE-resampled data) — the "Global Explainability" section (`summary_plot`, sampling
200–500 rows since computing SHAP for the whole dataset is very slow — the notebook notes "up to
30 minutes") and "Local Explainability" (`waterfall` per sample row). The SHAP explainer is
`joblib.dump`ed alongside the model for reuse in the deployed Streamlit app.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác `feature_importances_`/`gain` ([[xgboost]], [[random-forest]]) — những chỉ số đó là 1 con số
DUY NHẤT mỗi đặc trưng, tính trên toàn model; SHAP cho 1 giá trị RIÊNG cho MỖI (đặc trưng, dự
đoán), nên có thể tổng hợp thành cả góc nhìn global (trung bình |SHAP value| qua toàn bộ dữ liệu
≈ tương đương feature importance) LẪN local. Liên quan tới công bằng thuật toán ([[fairlearn]]) —
SHAP có thể dùng để kiểm tra xem model có dựa quá nhiều vào đặc trưng nhạy cảm (`is_male`,
`race_*`) cho MỘT NHÓM cụ thể hay không, bổ sung cho các chỉ số công bằng tổng hợp của FairLearn.
<br><span class="en">Different from `feature_importances_`/`gain` ([[xgboost]],
[[random-forest]]) — those are a SINGLE number per feature, computed across the whole model; SHAP
gives a SEPARATE value for EVERY (feature, prediction) pair, so it can be aggregated into both a
global view (average |SHAP value| across all data ≈ roughly equivalent to feature importance) AND
a local one. Related to algorithmic fairness ([[fairlearn]]) — SHAP can be used to check whether
the model relies too heavily on a sensitive feature (`is_male`, `race_*`) for a SPECIFIC group,
complementing FairLearn's aggregate fairness metrics.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] · [[xgboost]] ·
[[random-forest]] · [[fairlearn]]
