---
type: lecture
title: "WK2 — Data Preparation, Feature Engineering & Baseline Modelling"
tags: [data-preparation, feature-engineering, modelling, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk2-Data_Preparation_FeatureEngineering_Modelling.ipynb"
---

## Vị trí trong lộ trình môn học - <span class="en">Position in the Course</span>

Tiếp nối [[wk1-data-understanding]] — mọi phát hiện tuần 1 (dữ liệu thiếu ngụy trang, lệch
`native-country`, ý tưởng đặc trưng mới) nay được biến thành hành động cụ thể: làm sạch, tạo
đặc trưng, và huấn luyện 2 mô hình baseline đầu tiên. Kết quả yếu (F1 thấp trên lớp thiểu số) sẽ
dẫn thẳng tới [[wk3-modelling-imbalance-deep-learning]].
<br><span class="en">Follows directly from [[wk1-data-understanding]] — every Week-1 finding
(disguised missing data, `native-country` skew, new-feature ideas) now becomes concrete action:
cleaning, feature creation, and training the first two baseline models. The weak result (low F1
on the minority class) leads straight into
[[wk3-modelling-imbalance-deep-learning]].</span>

## Mục tiêu học tập - <span class="en">Learning Objectives</span>

Sau bài này, bạn làm được: (1) tự viết 1 sklearn `ColumnTransformer`+`Pipeline` hoàn chỉnh; (2)
dùng phân cụm phân cấp để gộp 1 biến phân loại nhiều mức thành vài nhóm có ý nghĩa; (3) giải
thích vì sao accuracy cao (83-84%) vẫn có thể là dấu hiệu XẤU khi dữ liệu mất cân bằng; (4) đọc
hiểu `feature_importances_` của Random Forest.
<br><span class="en">After this lesson you can: (1) build a complete sklearn
`ColumnTransformer`+`Pipeline` from scratch; (2) use hierarchical clustering to collapse a
high-cardinality categorical variable into a few meaningful groups; (3) explain why high accuracy
(83-84%) can still be a BAD sign when data is imbalanced; (4) read a Random Forest's
`feature_importances_`.</span>

## Lý thuyết nền tảng - <span class="en">Core Theory</span>

### 1. Data Preparation có 4 bước, không phải 1 bước "làm sạch" mơ hồ

CRISP-DM tách rõ: **Integrate** (gộp thêm nguồn dữ liệu khác, tuần này không có) → **Select**
(chọn dòng/cột nào giữ lại, tạo unique ID, tách train/test) → **Format & Clean** (chuẩn hóa kiểu
dữ liệu, xử lý dị thường) → **Construct** (tạo đặc trưng mới). Tách rõ 4 bước giúp không bỏ sót
— vd nhiều người chỉ nghĩ tới "Clean" mà quên "Construct" là bước RIÊNG BIỆT, độc lập.
<br><span class="en">CRISP-DM separates 4 distinct steps: **Integrate** (merge in other data
sources, none this week) → **Select** (choose which rows/columns to keep, create a unique ID,
split train/test) → **Format & Clean** (normalise data types, handle anomalies) → **Construct**
(create new features). Separating these 4 explicitly prevents gaps — e.g. many people only think
of "Clean" and forget "Construct" is a SEPARATE, independent step.</span>

### 2. Vì sao 1 ID tổ hợp (composite ID) có thể KHÔNG duy nhất

Thử tạo ID bằng `fnlwgt + age + education-num + hours-per-week + capital-gain` chỉ cho ra
44,089 giá trị duy nhất trên 48,842 dòng — nghĩa là có TRÙNG LẶP (2 dòng khác nhau hoàn toàn có
thể cho ra cùng tổng). Bài học: không bao giờ giả định 1 tổ hợp cột "trông có vẻ" duy nhất là
thực sự duy nhất — luôn kiểm tra bằng `.nunique()` so với `.shape[0]`, và dùng `uuid.uuid4()`
nếu cần 1 ID chắc chắn không đụng hàng.
<br><span class="en">Trying an ID built from `fnlwgt + age + education-num + hours-per-week +
capital-gain` yields only 44,089 unique values across 48,842 rows — meaning there ARE
collisions (two entirely different rows can sum to the same total). Lesson: never assume a
column combination that "looks" unique actually is — always verify with `.nunique()` versus
`.shape[0]`, and use `uuid.uuid4()` when you need a guaranteed-unique ID.</span>

### 3. Tách train/test PHẢI stratify khi target mất cân bằng

`train_test_split(..., stratify=Xy[TARGET_COL])` đảm bảo tỷ lệ `<=50K`/`>50K` ở tập train và
test GIỐNG HỆT tỷ lệ gốc (76.1%/23.9%) — nếu không stratify, phép chia ngẫu nhiên có thể vô
tình cho tập test nhiều/ít lớp thiểu số hơn thực tế, khiến điểm đánh giá bị sai lệch.
<br><span class="en">`train_test_split(..., stratify=Xy[TARGET_COL])` guarantees the
`<=50K`/`>50K` ratio in train and test EXACTLY MATCHES the original ratio (76.1%/23.9%) —
without stratifying, a random split could accidentally give the test set more/fewer minority
examples than reality, skewing the evaluation score.</span>

### 4. Nhóm biến phân loại nhiều mức bằng dữ liệu, không phải trực giác

Thay vì đoán mò cách gộp `occupation` (15 mức), notebook TÍNH tỷ lệ `>50K` cho từng mức rồi dùng
[[hierarchical-clustering-categorical-grouping]] để tìm các mức có hành vi tương tự nhau — cách
làm này khách quan hơn và có thể lặp lại được (reproducible) so với việc tự chọn nhóm theo cảm
tính.
<br><span class="en">Instead of guessing how to bucket `occupation` (15 levels), the notebook
COMPUTES the `>50K` rate for every level then uses
[[hierarchical-clustering-categorical-grouping]] to find levels with similar behaviour — this
approach is more objective and reproducible than manually choosing groups by intuition.</span>

### 5. Pipeline: đóng gói để không bao giờ "quên 1 bước" ở dữ liệu mới

[[sklearn-column-transformer-pipeline]] là điểm mấu chốt của cả tuần — thay vì viết từng dòng
biến đổi rời rạc (dễ quên áp lại cho test set), TOÀN BỘ logic tạo đặc trưng + chuẩn hóa được gói
vào 1 đối tượng `Pipeline` duy nhất, `fit` một lần trên train, `transform` lặp lại y hệt cho bất
kỳ dữ liệu mới nào (test set, hay dữ liệu thật khi deploy ở [[wk4-notebook]]).
<br><span class="en">[[sklearn-column-transformer-pipeline]] is the crux of the whole week —
instead of writing loose transformation statements (easy to forget re-applying to the test set),
ALL feature-creation + scaling logic is packaged into a single `Pipeline` object, `fit` once on
train, `transform`ed identically on any new data (the test set, or real data at deployment time
in [[wk4-notebook]]).</span>

### 6. Accuracy cao có thể là ảo giác khi dữ liệu mất cân bằng

Cả Random Forest và Logistic Regression đạt accuracy 83-84% — nghe có vẻ tốt, nhưng F1 trên lớp
`>50K` chỉ 0.64. Vì lớp `<=50K` chiếm 76%, một mô hình NGU NGỐC luôn đoán "`<=50K`" cho mọi
trường hợp đã đạt sẵn 76% accuracy mà không học được gì — con số 83-84% không cách xa mức nền
đó là bao. Đây là lý do WK3 chuyển hẳn sang đo bằng AUPRC.
<br><span class="en">Both Random Forest and Logistic Regression reach 83-84% accuracy —
sounds good, but F1 on the `>50K` class is only 0.64. Since `<=50K` is 76% of the data, a DUMB
model that always predicts "`<=50K`" already achieves 76% accuracy while learning nothing — the
83-84% figure isn't that far above that floor. This is why WK3 switches entirely to measuring
with AUPRC.</span>

## Ví dụ minh họa tổng hợp - <span class="en">Worked Example</span>

```python
import uuid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Select — unique id + stratified split
Xy['uuid'] = [uuid.uuid4() for _ in range(len(Xy))]
Train_Xy, Test_Xy = train_test_split(
    Xy, test_size=0.2, stratify=Xy['income'], random_state=42
)

# Feature Engineering — dùng lại pipeline từ [[sklearn-column-transformer-pipeline]]
Train_X_transformed = feature_engineering_pipeline.fit_transform(Train_X)
Test_X_transformed = feature_engineering_pipeline.transform(Test_X)   # KHÔNG fit lại

# Modelling — baseline
rf_model = RandomForestClassifier(random_state=42).fit(Train_X_transformed, Train_y_transformed)
print(classification_report(Test_y_transformed, rf_model.predict(Test_X_transformed)))
# accuracy 0.83, nhưng F1 lớp '>50K' chỉ 0.64 -> cần xử lý mất cân bằng lớp (WK3)
```

## Lỗi thường gặp & lưu ý thực hành - <span class="en">Pitfalls & Practical Notes</span>

- **`fit_transform` trên TOÀN BỘ dữ liệu (train+test gộp) rồi mới split** — rò rỉ thông tin từ
  test vào tham số biến đổi (vd `StandardScaler` học mean/std lẫn cả dữ liệu test); LUÔN split
  trước, `fit` chỉ trên train, `transform` (không fit) cho test.
  <br><span class="en">**`fit_transform` on ALL data (train+test combined) before splitting** —
  leaks test information into the transformation parameters (e.g. `StandardScaler` learning
  mean/std that include test data); ALWAYS split first, `fit` only on train, `transform`
  (without fitting) on test.</span>
- **Quên `stratify=` khi target mất cân bằng** — dẫn tới tỷ lệ lớp khác nhau giữa train/test,
  làm điểm đánh giá không đáng tin cậy.
  <br><span class="en">**Forgetting `stratify=` when the target is imbalanced** — leads to
  different class ratios between train/test, making the evaluation score unreliable.</span>
- **Chỉ nhìn accuracy mà bỏ qua F1/precision/recall theo từng lớp** — trên dữ liệu mất cân bằng,
  accuracy gần như luôn "trông ổn" một cách đánh lừa.
  <br><span class="en">**Only looking at accuracy while ignoring per-class F1/precision/recall**
  — on imbalanced data, accuracy almost always "looks fine" in a misleading way.</span>

## Liên kết concepts liên quan - <span class="en">Related Concepts</span>

[[sklearn-column-transformer-pipeline]] · [[hierarchical-clustering-categorical-grouping]] ·
[[random-forest]] · [[logistic-regression]]

## Câu hỏi tự kiểm tra nhanh - <span class="en">Quick Self-Check</span>

1. Vì sao phải `stratify=` khi tách train/test cho bài toán này? Điều gì xảy ra nếu bỏ qua?
   <br><span class="en">Why must you `stratify=` when splitting train/test for this problem?
   What happens if you skip it?</span>
2. `ColumnTransformer.fit()` nên gọi trên tập nào — train, test, hay cả hai gộp lại? Vì sao?
   <br><span class="en">Which set should `ColumnTransformer.fit()` be called on — train, test,
   or both combined? Why?</span>
3. Accuracy 83% nghe có vẻ tốt nhưng vì sao chưa đủ để kết luận mô hình tốt trong bài toán này?
   <br><span class="en">83% accuracy sounds good, but why isn't it enough to conclude the model
   is good for this problem?</span>

Bộ câu hỏi đầy đủ hơn: [[wk2-quiz]].
<br><span class="en">Full question bank: [[wk2-quiz]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-notebook]] · [[wk2-practice]] · [[wk1-data-understanding]] ·
[[wk3-modelling-imbalance-deep-learning]] · [[sklearn-column-transformer-pipeline]] ·
[[hierarchical-clustering-categorical-grouping]] · [[random-forest]] · [[logistic-regression]] ·
[[overview]]
