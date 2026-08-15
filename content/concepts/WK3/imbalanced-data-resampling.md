---
type: concept
title: "Xử Lý Mất Cân Bằng Lớp: SMOTE, ADASYN, Random Undersampling"
tags: [imbalanced-data, smote, adasyn, undersampling, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

3 kỹ thuật **resampling** để cân bằng lại tỷ lệ lớp trước khi huấn luyện: **SMOTE** (Synthetic
Minority Oversampling Technique — sinh mẫu thiểu số MỚI bằng nội suy), **ADASYN** (Adaptive
Synthetic Sampling — biến thể của SMOTE, sinh nhiều mẫu hơn ở vùng khó phân loại), và **Random
Undersampling** (bớt ngẫu nhiên mẫu đa số, không sinh gì mới).
<br><span class="en">3 **resampling** techniques to rebalance class ratios before training:
**SMOTE** (Synthetic Minority Oversampling Technique — generates NEW minority samples via
interpolation), **ADASYN** (Adaptive Synthetic Sampling — a SMOTE variant that generates more
samples in hard-to-classify regions), and **Random Undersampling** (randomly drops majority
samples, generates nothing new).</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Một mô hình huấn luyện trên dữ liệu mất cân bằng (lớp `>50K` chỉ 24%) có xu hướng thiên vị lớp đa
số, vì tối thiểu hóa loss tổng thể dễ dàng hơn bằng cách "đoán đại" lớp đa số cho các trường hợp
mơ hồ. Cân bằng lại tỷ lệ lớp trong TẬP HUẤN LUYỆN (không phải tập test) buộc mô hình phải học
đặc điểm phân biệt của lớp thiểu số thay vì bỏ qua nó.
<br><span class="en">A model trained on imbalanced data (the `>50K` class is only 24%) tends to
favour the majority class, because minimising overall loss is easier by "defaulting" to the
majority class for ambiguous cases. Rebalancing the class ratio in the TRAINING set (not the
test set) forces the model to learn the minority class's distinguishing features instead of
ignoring it.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

**SMOTE**: với mỗi mẫu thiểu số, tìm k láng giềng gần nhất CÙNG LỚP, chọn ngẫu nhiên 1 láng giềng,
tạo điểm mới nằm TRÊN ĐOẠN THẲNG nối 2 điểm đó (nội suy tuyến tính) — lặp lại tới khi số mẫu
thiểu số bằng số mẫu đa số. **ADASYN**: tương tự SMOTE nhưng PHÂN BỔ số mẫu sinh ra không đều —
sinh NHIỀU HƠN ở những mẫu thiểu số có nhiều láng giềng thuộc lớp đa số bao quanh (tức nằm ở vùng
biên, khó phân loại), ít hơn ở vùng thuần thiểu số (đã dễ phân loại). **Random Undersampling**:
chọn ngẫu nhiên 1 tập con của lớp đa số có kích thước bằng lớp thiểu số, GIỮ NGUYÊN toàn bộ lớp
thiểu số, loại bỏ phần còn lại của lớp đa số — không sinh dữ liệu mới nhưng có thể vứt bỏ thông
tin hữu ích.
<br><span class="en">**SMOTE**: for each minority sample, find its k nearest same-class
neighbours, randomly pick one, create a new point ON THE LINE SEGMENT joining the two (linear
interpolation) — repeat until minority count matches majority count. **ADASYN**: like SMOTE but
DISTRIBUTES the generated samples unevenly — generates MORE around minority samples surrounded by
many majority-class neighbours (i.e. in the hard-to-classify border region), fewer in purely
minority regions (already easy to classify). **Random Undersampling**: randomly selects a subset
of the majority class sized to match the minority class, KEEPS the entire minority class, drops
the rest of the majority — generates no new data but may discard useful information.</span>

## Ví dụ - <span class="en">Example</span>

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# SMOTE — mỗi kỹ thuật trả về (X, y) mới, đã cân bằng
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)
X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

print(y_train.value_counts())       # trước: 26510 vs 8555 (lệch)
print(pd.Series(y_smote).value_counts())  # sau SMOTE: 26510 vs 26510 (cân bằng)
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Trung tâm của [[wk3-notebook]]/"XGBoost for Imbalanced Data": SMOTE thắng rõ rệt (AUPRC ≈ 0.959)
so với ADASYN (≈0.949) và Undersampling trên bài toán này, được chọn cho bước tuning
[[hyperopt-tpe]] tiếp theo. [[wk4-notebook]] mở rộng thành 1 trong 6 chiến lược xử lý mất cân
bằng thử nghiệm trên cả lưới 9 model, cùng với [[ctgan]] (cơ chế sinh dữ liệu khác hẳn).
<br><span class="en">Central to [[wk3-notebook]]/"XGBoost for Imbalanced Data": SMOTE wins
clearly (AUPRC ≈ 0.959) over ADASYN (≈0.949) and Undersampling on this problem, and is chosen for
the following [[hyperopt-tpe]] tuning step. [[wk4-notebook]] expands this into one of 6
imbalance-handling strategies tested across the full 9-model grid, alongside [[ctgan]] (a
completely different generation mechanism).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

SMOTE vs ADASYN: cùng cơ chế nội suy tuyến tính, khác ở CÁCH PHÂN BỔ số mẫu sinh ra (đều vs tập
trung vùng biên) — ADASYN về lý thuyết giúp mô hình học tốt hơn ở vùng khó, nhưng trong bài toán
này SMOTE lại cho AUPRC cao hơn, nhắc nhở rằng "phức tạp hơn" không đồng nghĩa "tốt hơn" trên mọi
dữ liệu. Oversampling (SMOTE/ADASYN) vs Undersampling: oversampling giữ nguyên thông tin lớp đa
số nhưng tăng kích thước dữ liệu (chậm hơn khi huấn luyện); undersampling nhanh hơn nhưng vứt bỏ
dữ liệu. Khác hẳn về BẢN CHẤT với [[ctgan]] — SMOTE/ADASYN chỉ nội suy HÌNH HỌC giữa các điểm có
thật, còn CTGAN huấn luyện 1 mạng sinh (generator) để học lại toàn bộ phân phối dữ liệu.
<br><span class="en">SMOTE vs ADASYN: same linear-interpolation mechanism, differing in HOW
generated samples are DISTRIBUTED (even vs border-focused) — ADASYN theoretically helps the model
learn harder regions better, but on this problem SMOTE yields higher AUPRC, a reminder that "more
sophisticated" doesn't always mean "better" on every dataset. Oversampling (SMOTE/ADASYN) vs
Undersampling: oversampling preserves all majority-class information but grows the dataset
(slower training); undersampling is faster but discards data. Fundamentally different from
[[ctgan]] — SMOTE/ADASYN only interpolate GEOMETRICALLY between real points, while CTGAN trains a
generator network to relearn the entire data distribution.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[xgboost]] · [[auprc]] ·
[[hyperopt-tpe]] · [[ctgan]] · [[wk4-notebook]]
