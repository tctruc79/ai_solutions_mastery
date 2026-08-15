---
type: self-test
title: "WK2 Self-Test — Data Preparation & Feature Engineering"
tags: [self-test, wk2, feature-engineering]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Phạm vi: [[wk2-data-preparation-feature-engineering]], [[wk2-notebook]]. Câu hỏi tự luyện
Claude tự soạn, không phải đề thi chính thức.
<br><span class="en">Scope: [[wk2-data-preparation-feature-engineering]], [[wk2-notebook]].
Practice questions authored by Claude, not an official exam.</span>

## Câu 1 - <span class="en">Q1</span>

`fnlwgt + age + education-num + hours-per-week + capital-gain` chỉ cho ra 44,089 giá trị duy
nhất trên 48,842 dòng. Vấn đề này gọi là gì, và cách khắc phục đúng là gì?
<br><span class="en">`fnlwgt + age + education-num + hours-per-week + capital-gain` yields only
44,089 unique values out of 48,842 rows. What is this problem called, and what's the correct
fix?</span>

**Đáp án - <span class="en">Answer</span>**: Đây là collision (đụng độ) của một composite ID —
2 dòng dữ liệu khác nhau có thể cho cùng 1 tổng. Khắc phục: dùng `uuid.uuid4()` để sinh ID chắc
chắn duy nhất cho từng dòng, không dựa vào tổ hợp cột "trông có vẻ" duy nhất.
<br><span class="en">This is a collision in a composite ID — two different rows can produce the
same sum. Fix: use `uuid.uuid4()` to generate a guaranteed-unique ID per row, rather than relying
on a column combination that only "looks" unique.</span>

## Câu 2 - <span class="en">Q2</span>

Giải thích vì sao `ColumnTransformer.fit()` chỉ được gọi trên `Train_X`, không bao giờ trên
`Test_X` hay dữ liệu gộp cả hai.
<br><span class="en">Explain why `ColumnTransformer.fit()` should only be called on `Train_X`,
never on `Test_X` or the combined data.</span>

**Đáp án - <span class="en">Answer</span>**: Nếu `fit` trên cả test (hoặc dữ liệu gộp), các
tham số biến đổi (vd mean/std của `StandardScaler`) sẽ "nhìn thấy" thông tin từ test set — đây
là rò rỉ dữ liệu (data leakage), khiến điểm đánh giá trên test không còn phản ánh đúng hiệu năng
thực khi gặp dữ liệu hoàn toàn mới. Xem [[sklearn-column-transformer-pipeline]].
<br><span class="en">If you `fit` on test (or combined data), the transformation parameters
(e.g. `StandardScaler`'s mean/std) will "see" test-set information — this is data leakage,
making the test-set evaluation score no longer reflect true performance on genuinely new data.
See [[sklearn-column-transformer-pipeline]].</span>

## Câu 3 - <span class="en">Q3</span>

Vì sao dùng phân cụm phân cấp (hierarchical clustering) để nhóm `occupation` thay vì one-hot
toàn bộ 15 mức, hay tự chọn nhóm theo cảm tính?
<br><span class="en">Why use hierarchical clustering to group `occupation` instead of one-hot
encoding all 15 levels, or manually grouping by intuition?</span>

**Đáp án - <span class="en">Answer</span>**: One-hot toàn bộ tạo quá nhiều cột thưa, nhiều cột
gần như trùng hành vi với target; tự chọn nhóm theo cảm tính không khách quan/không lặp lại
được. Phân cụm dựa trên tỷ lệ `>50K` thực tế của từng mức cho ra nhóm KHÁCH QUAN, có cơ sở dữ
liệu, và vẫn giữ khả năng diễn giải (đặt tên nhóm được). Xem
[[hierarchical-clustering-categorical-grouping]].
<br><span class="en">Full one-hot creates too many sparse columns, many behaving almost
identically toward the target; manually choosing groups by intuition isn't objective/
reproducible. Clustering based on each level's actual `>50K` rate yields OBJECTIVE,
data-grounded groups that remain interpretable (nameable). See
[[hierarchical-clustering-categorical-grouping]].</span>

## Câu 4 - <span class="en">Q4</span>

Random Forest và Logistic Regression cùng đạt accuracy ~83-84% nhưng F1 lớp `>50K` chỉ 0.64. Vì
sao con số accuracy này "không ấn tượng như vẻ ngoài"?
<br><span class="en">Random Forest and Logistic Regression both reach ~83-84% accuracy but F1
on the `>50K` class is only 0.64. Why is this accuracy figure "less impressive than it
looks"?</span>

**Đáp án - <span class="en">Answer</span>**: Vì lớp `<=50K` chiếm 76.1% dữ liệu — một mô hình
NGU NGỐC luôn đoán `<=50K` cho mọi trường hợp đã đạt sẵn 76.1% accuracy mà không học được gì có
ích. 83-84% không cách xa mức nền đó nhiều, trong khi F1 = 0.64 trên lớp thiểu số cho thấy mô
hình thực sự còn yếu ở việc nhận diện lớp `>50K`. Xem [[wk2-data-preparation-feature-engineering]]
mục "Accuracy cao có thể là ảo giác".
<br><span class="en">Because the `<=50K` class is 76.1% of the data — a DUMB model that always
predicts `<=50K` already achieves 76.1% accuracy without learning anything useful. 83-84% isn't
far above that floor, while F1 = 0.64 on the minority class shows the model is genuinely still
weak at identifying `>50K`. See [[wk2-data-preparation-feature-engineering]], section "High
Accuracy Can Be an Illusion".</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-data-preparation-feature-engineering]] · [[wk2-notebook]] · [[wk1-quiz]] ·
[[cumulative-quiz]]
