---
type: concept
title: "ColumnTransformer + Pipeline Pattern"
tags: [sklearn, feature-engineering, pipeline, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

Mẫu thiết kế sklearn để gói toàn bộ bước biến đổi đặc trưng (tạo đặc trưng mới, chuẩn hóa, mã
hóa) thành **1 đối tượng duy nhất**, đảm bảo áp dụng chính xác cùng 1 phép biến đổi cho cả tập
huấn luyện và tập kiểm tra/dữ liệu mới.
<br><span class="en">A sklearn design pattern that bundles the entire feature-transformation
step (creating new features, scaling, encoding) into **a single object**, guaranteeing the exact
same transformation is applied to the training set and to the test set/new data.</span>

Gồm 3 thành phần lồng nhau: **custom transformer** (kế thừa `BaseEstimator, TransformerMixin`)
để tạo đặc trưng mới → **`ColumnTransformer`** để áp các phép biến đổi khác nhau cho từng nhóm
cột → **`Pipeline`** để nối các bước lại thành 1 quy trình `fit`/`transform` duy nhất.
<br><span class="en">Composed of 3 nested pieces: a **custom transformer** (subclassing
`BaseEstimator, TransformerMixin`) to create new features → **`ColumnTransformer`** to apply
different transforms to different column groups → **`Pipeline`** to chain the steps into one
`fit`/`transform` flow.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Nếu viết code biến đổi đặc trưng thành các dòng lệnh rời rạc (không đóng gói), rất dễ quên áp
dụng đúng cùng phép biến đổi cho tập test (vd quên `log1p` cho `capital-gain` ở test set), hoặc
vô tình "rò rỉ" thông tin từ test vào train khi `fit` scaler trên toàn bộ dữ liệu thay vì chỉ
trên train. Đóng gói vào 1 pipeline giải quyết cả 2 rủi ro này bằng thiết kế.
<br><span class="en">If feature-transformation code is written as loose, unpackaged
statements, it's easy to forget to apply the exact same transform to the test set (e.g.
forgetting `log1p` for `capital-gain` at test time), or to accidentally leak information from
test into train by fitting a scaler on the whole dataset instead of only on train. Bundling into
one pipeline eliminates both risks by design.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

`feature_engineering_pipeline.fit(Train_X)` chỉ học tham số (vd mean/std của `StandardScaler`)
từ TẬP TRAIN. `.transform(Test_X)` sau đó áp dụng lại đúng tham số đã học đó lên test — không
bao giờ để test set ảnh hưởng ngược lại tham số biến đổi.
<br><span class="en">`feature_engineering_pipeline.fit(Train_X)` learns parameters (e.g.
`StandardScaler`'s mean/std) ONLY from the TRAIN SET. `.transform(Test_X)` then reapplies those
exact learned parameters to test — the test set never influences the transformation parameters
back.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class CustomFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self                      # không học gì thêm, chỉ tạo cột mới có công thức cố định
    def transform(self, X):
        X = X.copy()
        X['is_married'] = (X['marital-status'] == 'Married-civ-spouse').astype(int)
        X['log_capital_gain'] = np.log1p(X['capital-gain'])
        return X

feature_transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'hours-per-week', 'log_capital_gain']),
    ('cat', 'drop', ['workclass', 'education', 'marital-status', ...]),
], remainder='passthrough')              # giữ nguyên các cờ 0/1 đã tạo

pipeline = Pipeline([
    ('feature_creation', CustomFeatureCreator()),
    ('feature_transform', feature_transformer),
])
pipeline.fit(Train_X)                    # học tham số CHỈ từ train
Train_X_transformed = pipeline.transform(Train_X)
Test_X_transformed = pipeline.transform(Test_X)   # áp dụng lại, KHÔNG fit lại
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Giới thiệu ở [[wk2-notebook]], **tái sử dụng nguyên vẹn** ở [[wk3-notebook]] và
[[wk4-notebook]] — cùng 1 hàm `build_features()` xuất hiện xuyên suốt 3 tuần, chỉ khác cách
dùng kết quả đầu ra (thêm resampling, thêm model khác nhau...).
<br><span class="en">Introduced in [[wk2-notebook]], **reused unchanged** in [[wk3-notebook]]
and [[wk4-notebook]] — the same `build_features()` function appears across 3 weeks, only how
the output is used downstream differs (adding resampling, different models...).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác `OneHotEncoder` áp dụng trực tiếp lên toàn bộ cột phân loại (không dùng trong pipeline này
— các cột phân loại thô bị `drop` sau khi thông tin của chúng đã được "nén" vào các cờ 0/1 thủ
công qua `CustomFeatureCreator`, một lựa chọn thiết kế có chủ đích thay vì one-hot tự động toàn
bộ, xem [[hierarchical-clustering-categorical-grouping]] để hiểu vì sao cách nhóm thủ công được
chọn thay one-hot).
<br><span class="en">Different from `OneHotEncoder` applied directly to all categorical columns
(not used in this pipeline — raw categorical columns are `drop`ped after their information has
been manually "compressed" into 0/1 flags via `CustomFeatureCreator`, a deliberate design choice
over automatic full one-hot encoding, see [[hierarchical-clustering-categorical-grouping]] to
understand why manual grouping was chosen over one-hot).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] ·
[[hierarchical-clustering-categorical-grouping]] · [[wk3-notebook]] · [[wk4-notebook]]
