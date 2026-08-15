---
type: concept
title: "Random Forest"
tags: [algorithm, ensemble, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**Random Forest** là mô hình ensemble gồm nhiều cây quyết định (decision tree) độc lập, mỗi cây
huấn luyện trên 1 mẫu bootstrap của dữ liệu và 1 tập con ngẫu nhiên của đặc trưng tại mỗi lần
chia nhánh; kết quả cuối là bỏ phiếu đa số (phân loại) hoặc trung bình (hồi quy) của toàn bộ cây.
<br><span class="en">**Random Forest** is an ensemble of many independent decision trees, each
trained on a bootstrap sample of the data and a random subset of features at every split; the
final result is a majority vote (classification) or average (regression) across all trees.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Một cây quyết định đơn lẻ rất dễ overfitting (khớp quá sát dữ liệu huấn luyện). Random Forest
giảm rủi ro này bằng cách lấy trung bình/bỏ phiếu của NHIỀU cây khác nhau (mỗi cây huấn luyện
trên dữ liệu/đặc trưng khác nhau một chút) — sai số riêng lẻ của từng cây có xu hướng triệt tiêu
lẫn nhau, còn tín hiệu thật thì được củng cố.
<br><span class="en">A single decision tree overfits easily (fits training data too closely).
Random Forest reduces this risk by averaging/voting across MANY different trees (each trained on
slightly different data/features) — individual trees' errors tend to cancel out, while genuine
signal gets reinforced.</span>

Được chọn làm mô hình baseline của môn học vì: xử lý tốt quan hệ phi tuyến, ít cần chuẩn hóa dữ
liệu đầu vào, cho ra `feature_importances_` miễn phí để diễn giải, và hoạt động tốt "out-of-box"
mà không cần tinh chỉnh nhiều.
<br><span class="en">Chosen as the course's baseline model because: it handles non-linear
relationships well, needs little input scaling, gives `feature_importances_` for free
interpretability, and performs well "out-of-box" without much tuning.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

`RandomForestClassifier(random_state=42)` mặc định dùng `n_estimators=100` cây, mỗi cây được
huấn luyện độc lập (có thể song song hóa). `.feature_importances_` sau khi fit trả về mức độ
đóng góp của mỗi đặc trưng vào việc giảm impurity trung bình qua toàn bộ cây.
<br><span class="en">`RandomForestClassifier(random_state=42)` defaults to `n_estimators=100`
trees, each trained independently (parallelisable). `.feature_importances_` after fitting
returns each feature's average contribution to impurity reduction across all trees.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(Train_X_transformed, Train_y_transformed)

importances = rf_model.feature_importances_       # 1 giá trị / 1 đặc trưng, tổng = 1.0
# feature quan trọng nhất tuần 2: 'binary__is_married' (~0.29)
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Giới thiệu làm baseline ở [[wk2-notebook]] (so với [[logistic-regression]]). WK3 đưa Random
Forest vào bảng so sánh 9 model (thường thua XGBoost/Gradient Boosting về AUPRC trên dữ liệu mất
cân bằng — xem [[wk3-notebook]]). WK4 chạy lại Random Forest trong grid 54 tổ hợp model×kỹ thuật
xử lý mất cân bằng.
<br><span class="en">Introduced as the baseline in [[wk2-notebook]] (versus
[[logistic-regression]]). WK3 adds Random Forest to the 9-model comparison table (typically
loses to XGBoost/Gradient Boosting on AUPRC with imbalanced data — see [[wk3-notebook]]). WK4
reruns Random Forest in the 54-combination model×imbalance-technique grid.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

So với [[logistic-regression]]: Random Forest bắt được quan hệ phi tuyến/tương tác giữa các đặc
trưng tốt hơn, nhưng khó diễn giải hệ số trực tiếp (không có "beta coefficient" như hồi quy
logistic) — 2 model cho kết quả accuracy/F1 gần tương đương ở WK2 (0.83 vs 0.84), khác biệt rõ
rệt hơn xuất hiện khi đo bằng AUPRC ở WK3. So với **XGBoost** ([[xgboost]], WK3) — cả hai đều là
ensemble cây, nhưng Random Forest xây cây ĐỘC LẬP song song (bagging) còn XGBoost xây cây TUẦN
TỰ, mỗi cây sửa lỗi của cây trước (boosting) — thường boosting cho AUPRC cao hơn trên bài toán
mất cân bằng của môn học này.
<br><span class="en">Versus [[logistic-regression]]: Random Forest captures non-linear
relationships/feature interactions better, but is harder to interpret directly (no "beta
coefficient" like logistic regression) — the two give near-identical accuracy/F1 in WK2 (0.83 vs
0.84), a clearer gap emerges once measured by AUPRC in WK3. Versus **XGBoost** ([[xgboost]],
WK3) — both are tree ensembles, but Random Forest builds trees INDEPENDENTLY in parallel
(bagging) while XGBoost builds trees SEQUENTIALLY, each correcting the previous one's errors
(boosting) — boosting typically wins on AUPRC for this course's imbalanced problem.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[logistic-regression]] ·
[[wk3-notebook]] · [[point-biserial-correlation]]
