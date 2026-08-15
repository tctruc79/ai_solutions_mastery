---
type: concept
title: "Point-Biserial Correlation"
tags: [statistics, correlation, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**Tương quan điểm-nhị phân (point-biserial correlation)** đo mức độ liên hệ tuyến tính giữa 1
biến liên tục và 1 biến nhị phân (chỉ 2 giá trị) — về mặt toán học, nó là một trường hợp đặc
biệt của tương quan Pearson khi 1 trong 2 biến được mã hóa 0/1.
<br><span class="en">**Point-biserial correlation** measures the linear relationship between a
continuous variable and a binary (2-value) variable — mathematically it is a special case of
Pearson correlation when one of the two variables is coded 0/1.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Tương quan Pearson chuẩn giả định cả 2 biến liên tục — không hợp lệ khi 1 biến là nhãn phân loại
như `income` (`<=50K`/`>50K`). Point-biserial giải quyết đúng trường hợp này mà không cần rời bỏ
khung tương quan Pearson quen thuộc.
<br><span class="en">Standard Pearson correlation assumes both variables are continuous — not
valid when one variable is a categorical label like `income` (`<=50K`/`>50K`). Point-biserial
correlation handles exactly this case without leaving the familiar Pearson framework.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

`scipy.stats.pointbiserialr(binary_var, continuous_var)` trả về hệ số r trong khoảng [-1, 1]
giống Pearson thông thường — dấu dương nghĩa là giá trị `1` của biến nhị phân đi kèm giá trị
biến liên tục cao hơn.
<br><span class="en">`scipy.stats.pointbiserialr(binary_var, continuous_var)` returns an r
coefficient in [-1, 1], interpreted exactly like ordinary Pearson r — a positive sign means the
binary variable's value `1` co-occurs with higher values of the continuous variable.</span>

## Ví dụ - <span class="en">Example</span>

```python
from scipy.stats import pointbiserialr

all_features['income_nr'] = (all_features['income'] == '>50K').astype(int)
corr, _ = pointbiserialr(all_features['income_nr'], all_features['age'])
# corr ≈ 0.178 — tuổi càng cao, xác suất thu nhập >50K càng nhích lên (yếu)
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk1-notebook]] mục 3.2 (áp dụng đầu tiên, so sánh `age`, `education-num`, `capital-gain`,
`capital-loss`, `hours-per-week` với `income`) — được [[wk1-practice]] tái sử dụng nguyên xi cho
bộ dữ liệu Bank Marketing (`duration`, `pdays`, `previous`, `campaign` với target `y`).
<br><span class="en">[[wk1-notebook]] section 3.2 (first application, comparing `age`,
`education-num`, `capital-gain`, `capital-loss`, `hours-per-week` against `income`) — reused
as-is by [[wk1-practice]] for the Bank Marketing dataset (`duration`, `pdays`, `previous`,
`campaign` against target `y`).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khi target có >2 lớp (không phải nhị phân), dùng **ANOVA** hoặc **Cramér's V** (cho 2 biến phân
loại) thay vì point-biserial — cả 2 kỹ thuật này được notebook gốc liệt kê nhưng không triển
khai chi tiết. Khi cần đo QUAN TRỌNG của biến trong một mô hình cụ thể (không chỉ tương quan đơn
lẻ) thì dùng feature importance của Random Forest ([[random-forest]]) hoặc [[shap]] — các kỹ
thuật này bắt được cả tương tác phi tuyến giữa nhiều biến mà point-biserial (chỉ xét 1-1, tuyến
tính) bỏ sót.
<br><span class="en">When the target has >2 classes (not binary), use **ANOVA** or **Cramér's
V** (for two categorical variables) instead of point-biserial — both are listed in the original
notebook but not implemented in detail. When you need to measure a variable's importance WITHIN
a specific model (not just standalone correlation), use Random Forest feature importance
([[random-forest]]) or [[shap]] — these capture non-linear interactions between multiple
variables that point-biserial (1-to-1, linear only) misses.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-notebook]] · [[wk1-data-understanding]] · [[wk1-practice]] · [[random-forest]] · [[shap]]
