---
type: concept
title: "CTGAN (Conditional Tabular GAN)"
tags: [ctgan, gan, synthetic-data, imbalanced-data, deep-learning, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**CTGAN** (Conditional Tabular GAN) là 1 kiến trúc **Generative Adversarial Network (GAN)** được
thiết kế riêng cho dữ liệu dạng bảng (tabular) — dùng để SINH dữ liệu tổng hợp giống phân phối dữ
liệu thật, trong môn học dùng để oversample lớp thiểu số cho bài toán Deep Learning.
<br><span class="en">**CTGAN** (Conditional Tabular GAN) is a **Generative Adversarial Network
(GAN)** architecture designed specifically for tabular data — used to GENERATE synthetic data
matching the real data distribution, applied in the course to oversample the minority class for
the Deep Learning problem.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

[[imbalanced-data-resampling]] (SMOTE/ADASYN) chỉ nội suy TUYẾN TÍNH giữa các điểm dữ liệu có
thật đã có — bị giới hạn trong "vùng lồi" (convex hull) của dữ liệu hiện tại, không thể tạo ra
điểm mới NGOÀI phạm vi đó. CTGAN thay vào đó HỌC một mô hình sinh (generator) đại diện cho toàn
bộ phân phối dữ liệu qua huấn luyện đối kháng — về lý thuyết có thể sinh mẫu đa dạng và chân thực
hơn, nắm bắt được các mối quan hệ phi tuyến phức tạp giữa nhiều cột mà nội suy tuyến tính bỏ lỡ.
<br><span class="en">[[imbalanced-data-resampling]] (SMOTE/ADASYN) only interpolates LINEARLY
between existing real data points — bounded within the current data's "convex hull", unable to
create new points OUTSIDE that range. CTGAN instead LEARNS a generator model representing the
entire data distribution through adversarial training — in principle it can produce more diverse
and realistic samples, capturing complex non-linear relationships across columns that linear
interpolation misses.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

Kiến trúc GAN gồm 2 mạng neural huấn luyện ĐỐI KHÁNG nhau: **Generator** cố sinh ra mẫu giả trông
giống dữ liệu thật; **Discriminator** cố phân biệt mẫu nào thật, mẫu nào giả — 2 mạng luân phiên
cải thiện nhau (Generator học cách đánh lừa Discriminator giỏi hơn, Discriminator học cách phát
hiện giỏi hơn) tới khi Generator sinh ra dữ liệu đủ chân thực. Khác các GAN ảnh thông thường,
CTGAN xử lý riêng cột liên tục (mô hình hóa bằng Gaussian mixture) và cột phân loại
(`discrete_columns=[...]`, dùng kỹ thuật training-by-sampling để tránh mode collapse trên cột
mất cân bằng). Trong môn học: `CTGAN(epochs=5).fit(df_train, discrete_columns=['target'])` rồi
`ctgan.sample(n)` sinh ra `n` mẫu tổng hợp mới.
<br><span class="en">The GAN architecture consists of 2 ADVERSARIALLY trained neural networks:
the **Generator** tries to produce fake samples that look like real data; the **Discriminator**
tries to tell real samples from fake ones — the two alternately improve each other (the Generator
learns to fool the Discriminator better, the Discriminator learns to detect better) until the
Generator produces sufficiently realistic data. Unlike typical image GANs, CTGAN handles
continuous columns separately (modelled via a Gaussian mixture) and categorical columns
(`discrete_columns=[...]`, using training-by-sampling to avoid mode collapse on imbalanced
columns). In the course: `CTGAN(epochs=5).fit(df_train, discrete_columns=['target'])` then
`ctgan.sample(n)` generates `n` new synthetic samples.</span>

## Ví dụ - <span class="en">Example</span>

```python
import pandas as pd
from ctgan import CTGAN

df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
df_train['target'] = y_train

ctgan = CTGAN(epochs=5)                              # rất ít epoch trong notebook gốc
ctgan.fit(df_train, discrete_columns=['target'])

# Sinh thêm mẫu bằng đúng số lượng mẫu lớp thiểu số hiện có
synthetic = ctgan.sample(len(y_train[y_train == 1]))
X_synthetic = synthetic.drop('target', axis=1).values
y_synthetic = synthetic['target'].values

import numpy as np
X_train_ctgan = np.vstack((X_train, X_synthetic))
y_train_ctgan = np.hstack((y_train, y_synthetic))
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Xuất hiện cuối [[wk3-notebook]]/"Deep Learning with Tensorflow/Keras" — so sánh Base Model vs
**CTGAN Oversampling** vs Undersampling qua 5-fold `StratifiedKFold`. Kết quả 1 fold minh họa:
CTGAN Oversampling (AUPRC ≈ 0.728) KHÔNG rõ ràng vượt trội Base Model (≈ 0.732) — output đầy đủ
5-fold không được lưu lại trong nguồn nên chưa thể kết luận chắc chắn. [[wk4-notebook]] mở rộng
CTGAN thành 1 trong 6 chiến lược xử lý mất cân bằng trên cả lưới 9 model — theo các trang liên
quan đã ghi nhận, XGBoost + CTGAN Oversampling thắng cuộc ở đó (AUPRC ≈ 0.964 trên CV, nhưng chỉ
~0.80 trên test set thật — 1 bài học quan trọng về đánh giá đúng cách khi dùng dữ liệu tổng hợp).
<br><span class="en">Appears at the end of [[wk3-notebook]]/"Deep Learning with
Tensorflow/Keras" — comparing Base Model vs **CTGAN Oversampling** vs Undersampling across a
5-fold `StratifiedKFold`. One illustrative fold: CTGAN Oversampling (AUPRC ≈ 0.728) does NOT
clearly outperform the Base Model (≈ 0.732) — the full 5-fold output wasn't preserved in the
source, so no firm conclusion can be drawn yet. [[wk4-notebook]] expands CTGAN into one of 6
imbalance strategies across the full 9-model grid — per related pages, XGBoost + CTGAN
Oversampling wins there (AUPRC ≈ 0.964 on CV, but only ~0.80 on the real test set — an important
lesson about correctly evaluating synthetic-data-trained models).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác [[imbalanced-data-resampling]] (SMOTE/ADASYN) về BẢN CHẤT cơ chế: SMOTE/ADASYN là phép tính
hình học TỨC THỜI (không cần huấn luyện gì), còn CTGAN là 1 MẠNG NEURAL cần huấn luyện (dù chỉ 5
epoch, vẫn tốn thời gian hơn nhiều, và cần lượng dữ liệu đủ lớn để học phân phối tốt). Đánh đổi:
SMOTE/ADASYN nhanh, đơn giản, dễ dự đoán hành vi; CTGAN chậm hơn, phức tạp hơn, nhưng tiềm năng
nắm bắt được cấu trúc dữ liệu phức tạp mà nội suy tuyến tính bỏ lỡ — tuy vậy kết quả thực nghiệm
trong môn học KHÔNG cho thấy CTGAN luôn thắng SMOTE (so [[wk3-notebook]] và ghi chú AUPRC
0.9590 của SMOTE+XGBoost so với ~0.728 của CTGAN+Deep Learning — dù 2 model nền khác nhau nên
không so trực tiếp được).
<br><span class="en">Fundamentally different mechanism from [[imbalanced-data-resampling]]
(SMOTE/ADASYN): SMOTE/ADASYN is an INSTANT geometric computation (no training needed), while
CTGAN is a NEURAL NETWORK requiring training (even at just 5 epochs, still far slower, and needs
enough data to learn the distribution well). Trade-off: SMOTE/ADASYN are fast, simple, predictable
in behaviour; CTGAN is slower, more complex, but potentially captures complex data structure that
linear interpolation misses — though the course's empirical results do NOT show CTGAN always
beating SMOTE (compare [[wk3-notebook]]'s SMOTE+XGBoost AUPRC of 0.9590 against CTGAN+Deep
Learning's ~0.728 — though the two use different base models, so they aren't directly
comparable).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[imbalanced-data-resampling]] ·
[[auprc]] · [[wk4-notebook]]
