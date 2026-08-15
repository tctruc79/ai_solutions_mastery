---
type: concept
title: "Logistic Regression"
tags: [algorithm, linear-model, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

Mô hình tuyến tính cho bài toán phân loại — ước lượng xác suất lớp dương bằng hàm sigmoid áp
lên tổ hợp tuyến tính của các đặc trưng đầu vào: `P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))`.
<br><span class="en">A linear model for classification — estimates the probability of the
positive class via a sigmoid function applied to a linear combination of input features:
`P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))`.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Đơn giản, huấn luyện nhanh, và mỗi hệ số `β` có thể diễn giải trực tiếp: dấu dương/âm cho biết
đặc trưng đó làm tăng/giảm xác suất lớp dương, độ lớn cho biết mức ảnh hưởng — đây là ưu điểm
Random Forest không có ([[random-forest]]). Vì mô hình tuyến tính, nó dùng làm đường cơ sở tốt
để đánh giá xem 1 model phức tạp hơn có thực sự đáng công sức thêm không.
<br><span class="en">Simple, fast to train, and every coefficient `β` is directly interpretable:
sign shows whether the feature increases/decreases the positive-class probability, magnitude
shows the strength — an advantage Random Forest lacks ([[random-forest]]). Because it's linear,
it makes a good baseline to judge whether a more complex model is actually worth the extra
effort.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

`LogisticRegression(random_state=42, max_iter=1000)` — `max_iter` tăng lên vì thuật toán tối ưu
(mặc định `lbfgs`) cần đủ vòng lặp để hội tụ trên dữ liệu đã chuẩn hóa (`StandardScaler`, xem
[[sklearn-column-transformer-pipeline]]) — chuẩn hóa đầu vào QUAN TRỌNG hơn nhiều với Logistic
Regression so với Random Forest, vì thuật toán tối ưu nhạy với thang đo của từng biến.
<br><span class="en">`LogisticRegression(random_state=42, max_iter=1000)` — `max_iter` is
raised because the optimiser (default `lbfgs`) needs enough iterations to converge on scaled
data (`StandardScaler`, see [[sklearn-column-transformer-pipeline]]) — input scaling matters
MUCH more for Logistic Regression than for Random Forest, since the optimiser is sensitive to
each variable's scale.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(Train_X_transformed, Train_y_transformed)
# accuracy 0.84, F1 lớp >50K = 0.64 — gần như ngang Random Forest ở tuần 2
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk2-notebook]] (baseline, đi kèm [[random-forest]]) → WK3 (1 trong 9 model so sánh, thêm
`class_weight='balanced'` để giảm nhẹ ảnh hưởng mất cân bằng lớp).
<br><span class="en">[[wk2-notebook]] (baseline, paired with [[random-forest]]) → WK3 (one of 9
compared models, adding `class_weight='balanced'` to soften class-imbalance effects).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Xem [[random-forest]] để so sánh trực tiếp (phi tuyến vs tuyến tính, diễn giải hệ số vs feature
importance).
<br><span class="en">See [[random-forest]] for a direct comparison (non-linear vs linear,
coefficient interpretation vs feature importance).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[random-forest]] ·
[[sklearn-column-transformer-pipeline]]
