---
type: concept
title: "AUPRC (Area Under the Precision-Recall Curve)"
tags: [metric, evaluation, imbalanced-data, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**AUPRC** (cũng gọi AUC-PR, hay `average_precision_score` trong sklearn) là diện tích dưới
đường cong Precision-Recall — đo khả năng của mô hình giữ được cả precision (độ chính xác, ít
báo sai) LẪN recall (độ bao phủ, ít bỏ sót) cùng lúc, tổng hợp qua mọi ngưỡng phân loại có thể.
<br><span class="en">**AUPRC** (also called AUC-PR, or `average_precision_score` in sklearn) is
the area under the Precision-Recall curve — measures a model's ability to hold both precision
(few false alarms) AND recall (few misses) simultaneously, aggregated across every possible
classification threshold.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Trên dữ liệu mất cân bằng (lớp `>50K` chỉ 23.9%), **accuracy gây hiểu lầm** (xem
[[wk2-data-preparation-feature-engineering]]) và ngay cả **ROC-AUC** cũng có thể "trông tốt" một
cách ảo vì đường ROC bị chi phối bởi True Negative Rate — với lớp âm chiếm đa số áp đảo, TNR gần
như luôn cao dù mô hình dở ở lớp thiểu số. AUPRC tập trung HOÀN TOÀN vào lớp dương (thiểu số),
nên phản ánh đúng hơn hiệu năng thực trên bài toán mất cân bằng.
<br><span class="en">On imbalanced data (the `>50K` class is only 23.9%), **accuracy is
misleading** (see [[wk2-data-preparation-feature-engineering]]) and even **ROC-AUC** can look
deceptively good because the ROC curve is dominated by the True Negative Rate — with the
negative class an overwhelming majority, TNR stays high almost regardless of how bad the model
is on the minority class. AUPRC focuses ENTIRELY on the positive (minority) class, so it more
honestly reflects real performance on an imbalanced problem.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

Với mỗi ngưỡng phân loại có thể, tính (Precision, Recall) tương ứng → nối các điểm đó thành 1
đường cong → AUPRC = diện tích dưới đường cong đó. Baseline ngẫu nhiên của AUPRC (không phải
0.5 như ROC-AUC) BẰNG TỶ LỆ lớp dương trong dữ liệu — vd với lớp `>50K` = 23.9%, một mô hình
đoán ngẫu nhiên hoàn toàn chỉ đạt AUPRC ≈ 0.239, nên phải so sánh AUPRC với con số nền này chứ
không phải với 0.5 hay 1.0.
<br><span class="en">For every possible classification threshold, compute the corresponding
(Precision, Recall) → connect those points into a curve → AUPRC = area under that curve.
AUPRC's random baseline (unlike ROC-AUC's 0.5) EQUALS the positive-class rate in the data —
e.g. with `>50K` = 23.9%, a purely random model only reaches AUPRC ≈ 0.239, so AUPRC must be
compared against this floor, not against 0.5 or 1.0.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

auprc_scorer = make_scorer(average_precision_score, needs_proba=True)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring=auprc_scorer)
print(scores.mean())   # so sánh với baseline ngẫu nhiên ≈ tỷ lệ lớp dương (0.239), KHÔNG phải 0.5
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Trở thành thước đo CHÍNH THỨC từ [[wk3-notebook]] trở đi (so sánh 9 model, tuning XGBoost với
[[hyperopt-tpe]], grid 54 tổ hợp ở [[wk4-notebook]]). Bài học quan trọng ở
[[wk4-notebook]]/Part IV: AUPRC đo trên dữ liệu ĐÃ resample (SMOTE/CTGAN) bị lạc quan giả — phải
luôn kiểm tra lại trên tập test THẬT chưa resample.
<br><span class="en">Becomes the OFFICIAL metric from [[wk3-notebook]] onward (comparing 9
models, tuning XGBoost with [[hyperopt-tpe]], the 54-combination grid in [[wk4-notebook]]). A
key lesson in [[wk4-notebook]]/Part IV: AUPRC measured on RESAMPLED data (SMOTE/CTGAN) is
falsely optimistic — always re-check on the REAL, unresampled test set.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác **ROC-AUC** — ROC-AUC dùng cả TPR và FPR (bị pha loãng bởi lớp âm áp đảo), còn AUPRC dùng
Precision và Recall (cả hai đều tính TRÊN lớp dương, không bị pha loãng bởi số lượng lớp âm khổng
lồ) → AUPRC nhạy hơn nhiều với hiệu năng thực trên lớp thiểu số. Khác **F1-score** — F1 chỉ là 1
điểm tại 1 ngưỡng cố định (thường 0.5), còn AUPRC tổng hợp qua MỌI ngưỡng, không phụ thuộc việc
chọn ngưỡng cụ thể.
<br><span class="en">Different from **ROC-AUC** — ROC-AUC uses both TPR and FPR (diluted by the
overwhelming negative class), while AUPRC uses Precision and Recall (both computed ON the
positive class, not diluted by the huge negative-class count) → AUPRC is far more sensitive to
real minority-class performance. Different from **F1-score** — F1 is a single point at one fixed
threshold (usually 0.5), while AUPRC aggregates across EVERY threshold, independent of any
specific threshold choice.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] ·
[[wk2-data-preparation-feature-engineering]] · [[xgboost]] · [[imbalanced-data-resampling]]
