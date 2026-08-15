---
type: concept
title: "CRISP-DM"
tags: [methodology, crisp-dm, workflow]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) là quy trình 6 giai đoạn dùng để
tổ chức một dự án khoa học dữ liệu từ đầu đến cuối.
<br><span class="en">**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is a
6-phase process for structuring a data-science project end-to-end.</span>

Toàn bộ 4 tuần của môn học bám sát đúng thứ tự này — mỗi notebook lecture dùng CHÍNH các heading
markdown trùng tên giai đoạn CRISP-DM.
<br><span class="en">All 4 weeks of the course follow this exact order — each lecture notebook
uses markdown headings that literally match the CRISP-DM phase names.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Nếu nhảy thẳng vào việc huấn luyện mô hình mà bỏ qua bước hiểu và làm sạch dữ liệu, mô hình sẽ
học đúng theo lỗi/thiên lệch có sẵn trong dữ liệu (garbage in, garbage out).
<br><span class="en">Jumping straight to model training while skipping the data-understanding
and cleaning steps means the model faithfully learns whatever errors/biases already exist in the
data (garbage in, garbage out).</span>

## Cơ chế hoạt động - <span class="en">The 6 Phases</span>

1. **Business Understanding** — hiểu mục tiêu kinh doanh/bài toán (môn học không tách riêng
   phần này thành mục lớn, nhưng luôn nêu ngắn ở đầu mỗi notebook, vd bài toán dự đoán thu nhập
   >\$50K).
   <br><span class="en">**Business Understanding** — understand the business goal/problem (the
   course doesn't dedicate a separate large section to this, but always states it briefly at the
   top of each notebook, e.g. the >\$50K income prediction task).</span>
2. **Data Understanding** = WK1 của môn này — Collect → Describe → Explore → Verify Data
   Quality, xem [[wk1-data-understanding]].
   <br><span class="en">**Data Understanding** = this course's WK1 — Collect → Describe →
   Explore → Verify Data Quality, see [[wk1-data-understanding]].</span>
3. **Data Preparation** = phần đầu WK2 — Integrate → Select → Format & Clean → Construct Data,
   xem [[wk2-data-preparation-feature-engineering]].
   <br><span class="en">**Data Preparation** = the first half of WK2 — Integrate → Select →
   Format & Clean → Construct Data, see [[wk2-data-preparation-feature-engineering]].</span>
4. **Modelling** = phần sau WK2 + toàn bộ WK3 + phần đầu WK4 — chọn kỹ thuật, xây model, đánh
   giá, chọn đặc trưng.
   <br><span class="en">**Modelling** = the second half of WK2 + all of WK3 + the first half of
   WK4 — select technique, build model, assess, feature selection.</span>
5. **Evaluation** — trong môn này lồng vào bước Model Selection ở WK4 (kiểm tra lại trên tập
   test THẬT thay vì tin số liệu cross-validation trên dữ liệu đã resample) — xem
   [[wk4-model-selection-explainability-fairness-deployment]].
   <br><span class="en">**Evaluation** — folded into WK4's Model Selection step in this course
   (re-checking on the REAL test set rather than trusting cross-validation numbers computed on
   resampled data) — see [[wk4-model-selection-explainability-fairness-deployment]].</span>
6. **Deployment** = cuối WK4 — lưu model bằng `joblib`, khung Streamlit app.
   <br><span class="en">**Deployment** = the end of WK4 — persisting the model with `joblib`, a
   Streamlit app scaffold.</span>

## Ví dụ - <span class="en">Example</span>

```python
# Giai đoạn Data Understanding (WK1) luôn bắt đầu bằng shape + describe, KHÔNG build model vội
input_features.shape        # (48842, 14)
input_features.describe()   # thống kê mô tả trước khi quyết định bất kỳ xử lý nào
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Toàn bộ 4 tuần: [[wk1-notebook]], [[wk2-notebook]], [[wk3-notebook]], [[wk4-notebook]].
<br><span class="en">All 4 weeks: [[wk1-notebook]], [[wk2-notebook]], [[wk3-notebook]],
[[wk4-notebook]].</span>

## So sánh với quy trình liên quan - <span class="en">Related Frameworks</span>

CRISP-DM khác **MLOps lifecycle** (thêm CI/CD, monitoring liên tục sau deploy — môn học chỉ dừng
ở khung Streamlit, chưa vào MLOps sâu) và khác **OSEMN** (Obtain-Scrub-Explore-Model-iNterpret,
một biến thể gọn hơn, ít nhấn mạnh Business Understanding).
<br><span class="en">CRISP-DM differs from the **MLOps lifecycle** (adds CI/CD, continuous
post-deployment monitoring — this course only reaches a Streamlit scaffold, not deep MLOps) and
from **OSEMN** (Obtain-Scrub-Explore-Model-iNterpret, a leaner variant that de-emphasises
Business Understanding).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[overview]] · [[wk1-data-understanding]] · [[wk2-data-preparation-feature-engineering]] ·
[[wk4-model-selection-explainability-fairness-deployment]]
