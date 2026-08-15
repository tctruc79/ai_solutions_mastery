---
type: self-test
title: "WK1 Self-Test — Data Understanding"
tags: [self-test, wk1, data-understanding]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Phạm vi: [[wk1-data-understanding]], [[wk1-notebook]], [[wk1-practice]]. Câu hỏi tự luyện
Claude tự soạn, không phải đề thi chính thức.
<br><span class="en">Scope: [[wk1-data-understanding]], [[wk1-notebook]], [[wk1-practice]].
Practice questions authored by Claude, not an official exam.</span>

## Câu 1 - <span class="en">Q1</span>

`target_feature['income'].value_counts()` cho ra 4 giá trị khác nhau: `<=50K`, `<=50K.`,
`>50K`, `>50K.`. Giải thích vì sao có 4 giá trị thay vì 2, và viết 1 dòng code để gộp về đúng 2
nhãn.
<br><span class="en">`target_feature['income'].value_counts()` returns 4 distinct values:
`<=50K`, `<=50K.`, `>50K`, `>50K.`. Explain why there are 4 values instead of 2, and write one
line of code to collapse them into exactly 2 labels.</span>

**Đáp án - <span class="en">Answer</span>**: UCI Adult gồm 2 phần train/test được gộp chung;
phần test có dấu `.` ở cuối nhãn còn phần train thì không → cùng 1 ý nghĩa nhưng bị đếm là 4 giá
trị riêng biệt. Xử lý: `target_feature['income'].str.replace('.', '', regex=False).str.strip()`
(xem [[wk1-notebook]] mục 4.1).
<br><span class="en">UCI Adult combines a train and a test portion; the test portion has a
trailing `.` while train doesn't → same meaning but counted as 4 distinct values. Fix:
`target_feature['income'].str.replace('.', '', regex=False).str.strip()` (see [[wk1-notebook]]
section 4.1).</span>

## Câu 2 - <span class="en">Q2</span>

Vì sao không thể áp dụng tương quan Pearson tiêu chuẩn trực tiếp giữa `hours-per-week` và
`income` (dạng chữ)? Kỹ thuật nào nên dùng thay thế, và tên hàm scipy tương ứng?
<br><span class="en">Why can't standard Pearson correlation be applied directly between
`hours-per-week` and `income` (text form)? What technique should be used instead, and what is
the corresponding scipy function?</span>

**Đáp án - <span class="en">Answer</span>**: Pearson giả định cả 2 biến liên tục; `income` là
biến phân loại (dù chỉ 2 mức). Cần nhị phân hóa `income` (0/1) rồi dùng **point-biserial
correlation** — `scipy.stats.pointbiserialr()`. Xem [[point-biserial-correlation]].
<br><span class="en">Pearson assumes both variables are continuous; `income` is categorical
(even though only 2 levels). You must binarise `income` (0/1) then use **point-biserial
correlation** — `scipy.stats.pointbiserialr()`. See [[point-biserial-correlation]].</span>

## Câu 3 - <span class="en">Q3</span>

`input_features.isnull().sum()` báo giá trị 0 cho toàn bộ 14 cột, nhưng `workclass` thực tế có
1,836 dòng thiếu dữ liệu. Vì sao? Nêu 1 cách kiểm tra chính xác hơn `isnull()` trong trường hợp
này.
<br><span class="en">`input_features.isnull().sum()` reports 0 for all 14 columns, yet
`workclass` actually has 1,836 missing rows. Why? Name a check that's more reliable than
`isnull()` in this case.</span>

**Đáp án - <span class="en">Answer</span>**: Giá trị thiếu được mã hóa bằng ký tự `'?'`, không
phải `NaN` chuẩn, nên `isnull()` không phát hiện được. Kiểm tra chính xác hơn:
`input_features['workclass'].value_counts()` — sẽ thấy `'?'` xuất hiện như 1 "category". Xem
[[wk1-data-understanding]] mục "Dữ liệu thiếu không phải lúc nào cũng là NaN".
<br><span class="en">Missing values are encoded with the character `'?'`, not standard `NaN`, so
`isnull()` doesn't catch them. A more reliable check:
`input_features['workclass'].value_counts()` — `'?'` will show up as a "category". See
[[wk1-data-understanding]], section "Missing Data Isn't Always NaN".</span>

## Câu 4 - <span class="en">Q4</span>

Trong bài assignment (Bank Marketing), cột `poutcome` thiếu tới 81.7%. Vì sao KHÔNG nên impute
(điền giá trị thay thế) cho cột này theo cách thông thường?
<br><span class="en">In the assignment (Bank Marketing), the `poutcome` column is 81.7%
missing. Why should this column NOT be imputed the usual way?</span>

**Đáp án - <span class="en">Answer</span>**: Đây là **missing có tính cấu trúc (structural
missingness)** — khách hàng thiếu `poutcome` đơn giản vì họ CHƯA TỪNG được liên hệ ở chiến dịch
trước, không phải dữ liệu bị mất/lỗi thu thập. Cách đúng: giữ nguyên `'unknown'`/mã hóa thành 1
category riêng (hoặc cờ boolean `no_previous_contact`), không impute bằng mode/mean vì điều đó
sẽ xóa đi chính thông tin "chưa từng liên hệ" — xem [[wk1-practice]].
<br><span class="en">This is **structural missingness** — clients are missing `poutcome`
simply because they were NEVER contacted in a previous campaign, not because of a data
collection error. Correct treatment: keep `'unknown'` / encode it as its own category (or a
`no_previous_contact` boolean flag), do not impute with mode/mean since that would erase exactly
the "never contacted" information — see [[wk1-practice]].</span>

## Câu 5 - <span class="en">Q5</span>

Nêu 1 lý do vì sao nên đặt câu hỏi về công bằng thuật toán (fairness) ngay từ giai đoạn khám phá
dữ liệu, thay vì đợi tới khi có mô hình.
<br><span class="en">Give one reason why fairness questions should be raised at the
data-exploration stage rather than waiting until a model exists.</span>

**Đáp án - <span class="en">Answer</span>**: Skew nhân khẩu học trong dữ liệu gốc (vd Nam
66.85%/Nữ 33.15%, `White` 85.5%) là NGUỒN GỐC của thiên lệch mô hình về sau; phát hiện càng sớm
thì càng nhiều lựa chọn xử lý (thu thập thêm dữ liệu cân bằng hơn, đặt trọng số mẫu, ràng buộc
công bằng khi huấn luyện) — đợi tới khi mô hình đã huấn luyện xong thì lựa chọn còn lại chủ yếu
là hậu xử lý (post-processing), hạn chế hơn nhiều. Xem [[fairlearn]] (WK4) để biết chi tiết các
kỹ thuật xử lý.
<br><span class="en">Demographic skew in the raw data (e.g. Male 66.85%/Female 33.15%, `White`
85.5%) is the ROOT CAUSE of later model bias; catching it earlier leaves more remediation
options (collect more balanced data, sample weighting, fairness constraints during training) —
wait until after a model is trained and the remaining options are mostly post-processing, far
more limited. See [[fairlearn]] (WK4) for remediation technique details.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-data-understanding]] · [[wk1-notebook]] · [[wk1-practice]] · [[cumulative-quiz]]
