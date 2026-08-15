---
type: lecture
title: "WK1 — Data Understanding"
tags: [data-understanding, crisp-dm, eda, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai_practitioner_wk1_Data_Understanding.ipynb"
---

## Vị trí trong lộ trình môn học - <span class="en">Position in the Course</span>

Đây là tuần đầu tiên, giai đoạn **Data Understanding** của [[crisp-dm]] — trước khi chạm vào bất
kỳ mô hình nào, phải hiểu dữ liệu đang có gì, thiếu gì, lệch ở đâu. Tuần 2 sẽ dùng trực tiếp mọi
phát hiện ở đây (đặc biệt là quyết định lọc `native-country == 'United-States'` và ý tưởng đặc
trưng mới) để chuẩn bị dữ liệu cho mô hình.
<br><span class="en">This is the first week, the **Data Understanding** phase of [[crisp-dm]] —
before touching any model, you must understand what the data has, what's missing, where it's
skewed. Week 2 directly reuses every finding here (especially the decision to filter to
`native-country == 'United-States'` and the new-feature ideas) to prepare the data for
modelling.</span>

## Mục tiêu học tập - <span class="en">Learning Objectives</span>

Sau bài này, bạn làm được: (1) tải dữ liệu từ UCI Machine Learning Repository bằng
`ucimlrepo`; (2) phân biệt được khi nào dùng tương quan Pearson thường và khi nào cần
point-biserial; (3) nhận diện dữ liệu thiếu bị "ngụy trang" (không phải lúc nào cũng là `NaN`);
(4) tự đặt câu hỏi công bằng thuật toán (fairness) ngay từ bước khám phá dữ liệu, trước khi có
mô hình nào cả.
<br><span class="en">After this lesson you can: (1) load data from the UCI Machine Learning
Repository via `ucimlrepo`; (2) tell when to use ordinary Pearson correlation versus
point-biserial correlation; (3) recognise "disguised" missing data (not always literal `NaN`);
(4) ask fairness questions at the data-exploration stage already, before any model exists.</span>

## Lý thuyết nền tảng - <span class="en">Core Theory</span>

### 1. Vì sao Data Understanding đứng trước Modelling

Một mô hình chỉ tốt bằng dữ liệu nó được huấn luyện. Nếu bỏ qua bước hiểu dữ liệu, những vấn đề
như giá trị thiếu bị mã hóa lạ (`'?'` thay vì `NaN`), nhãn target không nhất quán (`>50K` vs
`>50K.`), hay dữ liệu lệch nhân khẩu học sẽ âm thầm lọt vào mô hình và bị hiểu nhầm là "tín hiệu
thật" thay vì lỗi dữ liệu.
<br><span class="en">A model is only as good as the data it's trained on. Skip the
data-understanding step and problems like oddly-encoded missing values (`'?'` instead of
`NaN`), inconsistent target labels (`>50K` vs `>50K.`), or demographic skew silently leak into
the model and get mistaken for a real signal instead of a data defect.</span>

### 2. Tải dữ liệu từ UCI qua `ucimlrepo`

Cách chuẩn: vào trang UCI Machine Learning Repository của bộ dữ liệu, bấm "Import in Python",
copy code sinh sẵn.
<br><span class="en">Standard approach: go to the UCI Machine Learning Repository page for the
dataset, click "Import in Python", copy the generated code.</span>

```python
from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)              # id=2 là mã riêng của bộ Adult Census Income
input_features = adult.data.features     # DataFrame 48,842 × 14
target_feature = adult.data.targets      # DataFrame 48,842 × 1, cột 'income'
print(adult.metadata['abstract'])        # mô tả bài toán ngay từ nguồn
```

### 3. Hai kiểu tương quan cho hai kiểu bài toán

Khi biến X liên tục và biến Y cũng liên tục → Pearson chuẩn (`df.corr()`). Khi biến X liên tục
còn biến Y là target NHỊ PHÂN (như `income`) → **point-biserial** (xem
[[point-biserial-correlation]]), KHÔNG dùng Pearson trực tiếp trên nhãn chữ.
<br><span class="en">When X is continuous and Y is also continuous → ordinary Pearson
(`df.corr()`). When X is continuous but Y is a BINARY target (like `income`) → **point-biserial**
(see [[point-biserial-correlation]]), do NOT run Pearson directly on a string label.</span>

```python
from scipy.stats import pointbiserialr

y_binary = (target_feature['income'] == '>50K').astype(int)
r, _ = pointbiserialr(y_binary, input_features['age'])   # r ≈ 0.18
```

### 4. Dữ liệu thiếu không phải lúc nào cũng là `NaN`

`input_features.isnull().sum()` cho ra 0 với `workclass`, nhưng thực tế cột này CÓ 1,836 dòng
thiếu — chỉ là chúng được mã hóa bằng ký tự `'?'`. Đây là bài học quan trọng nhất của tuần: LUÔN
kiểm tra `value_counts()` của cột phân loại trước khi tin vào `isnull()`.
<br><span class="en">`input_features.isnull().sum()` reports 0 for `workclass`, but the column
actually HAS 1,836 missing rows — they're just encoded as the character `'?'`. This is the
week's most important lesson: ALWAYS inspect `value_counts()` on categorical columns before
trusting `isnull()`.</span>

```python
input_features['workclass'].value_counts()
# ...
# ?                    1836      <- đây chính là giá trị thiếu, không phải 1 category thật
# Never-worked            10
```

### 5. Nguyên tắc "không impute mù quáng"

Notebook gốc nêu rõ: không được tự động điền `0`/trung bình cho giá trị thiếu mà không điều tra
CƠ CHẾ thiếu dữ liệu trước — dữ liệu thiếu có thể **structural** (thiếu vì logic, vd
`poutcome='unknown'` khi khách chưa từng được liên hệ trước đó — xem [[wk1-practice]]) hoặc thật
sự ngẫu nhiên. Cách xử lý đúng khác nhau hoàn toàn giữa 2 trường hợp.
<br><span class="en">The original notebook states clearly: never auto-fill missing values with
0/mean without first investigating the MECHANISM of missingness — data can be missing
**structurally** (missing by logic, e.g. `poutcome='unknown'` when a client was never previously
contacted — see [[wk1-practice]]) or genuinely at random. The correct treatment differs
completely between the two cases.</span>

### 6. Đặt câu hỏi công bằng (fairness) NGAY TỪ khâu khám phá dữ liệu

Trước khi huấn luyện bất kỳ mô hình nào, notebook đã ghi nhận: Nam 66.85% / Nữ 33.15%, và
`White` chiếm 85.50% — lệch đáng kể so với thực tế dân số Mỹ. Đây là tín hiệu cảnh báo sớm: một
mô hình huấn luyện trên dữ liệu này CÓ THỂ học được thiên lệch giới tính/chủng tộc, và câu hỏi
công bằng cần được đặt ra từ bây giờ chứ không đợi đến khi có mô hình xong (chủ đề này quay lại
đầy đủ ở [[fairlearn]], tuần 4).
<br><span class="en">Before training any model, the notebook already notes: Male 66.85% /
Female 33.15%, and `White` at 85.50% — a meaningful skew relative to actual US population
demographics. This is an early warning sign: a model trained on this data COULD learn
gender/race bias, and the fairness question should be raised now, not after a model already
exists (this topic returns in full in [[fairlearn]], Week 4).</span>

## Ví dụ minh họa tổng hợp - <span class="en">Worked Example</span>

Chuỗi thao tác tối thiểu để đi từ tải dữ liệu thô đến một target sạch, sẵn sàng cho tuần 2:
<br><span class="en">The minimal sequence from raw data load to a clean target, ready for Week
2:</span>

```python
from ucimlrepo import fetch_ucirepo
from scipy.stats import pointbiserialr
import pandas as pd

# 1. Collect
adult = fetch_ucirepo(id=2)
X, y = adult.data.features, adult.data.targets

# 2. Describe
print(X.shape, y.shape)                          # (48842, 14) (48842, 1)
print(X.isnull().sum())                           # cẩn thận: '?' KHÔNG hiện ở đây

# 3. Explore — tương quan điểm-nhị phân, không phải Pearson thường
y_bin = (y['income'].str.contains('>50K')).astype(int)
for col in ['age', 'education-num', 'hours-per-week']:
    r, _ = pointbiserialr(y_bin, X[col])
    print(col, round(r, 3))

# 4. Verify Data Quality — chuẩn hóa target về đúng 2 nhãn
y['income'] = y['income'].str.replace('.', '', regex=False).str.strip()
print(y['income'].value_counts())                 # đúng 2 nhãn: <=50K / >50K
```

## Lỗi thường gặp & lưu ý thực hành - <span class="en">Pitfalls & Practical Notes</span>

- **Tin `isnull().sum() == 0` mà không kiểm tra `value_counts()`** — dữ liệu thực tế hay mã hóa
  missing bằng ký tự đặc biệt (`'?'`, `'unknown'`, `-1`, `999`...) tùy nguồn.
  <br><span class="en">**Trusting `isnull().sum() == 0` without checking `value_counts()`** —
  real-world data often encodes missingness with a special token (`'?'`, `'unknown'`, `-1`,
  `999`...) depending on the source.</span>
- **Chạy Pearson trên cột string** — sẽ lỗi hoặc cho kết quả vô nghĩa; phải nhị phân hóa target
  trước rồi mới dùng point-biserial.
  <br><span class="en">**Running Pearson on a string column** — will error or produce
  meaningless results; binarise the target first, then use point-biserial.</span>
- **Bỏ qua bước fairness vì "chưa có model"** — càng sớm càng tốt, vì skew trong dữ liệu gốc là
  nguồn gốc của thiên lệch mô hình sau này, phát hiện sớm thì phương án xử lý (thêm dữ liệu, đặt
  trọng số mẫu, ràng buộc công bằng) cũng nhiều lựa chọn hơn.
  <br><span class="en">**Skipping fairness because "there's no model yet"** — the earlier the
  better, since skew in the raw data is the root cause of later model bias; catching it early
  leaves more remediation options (collect more data, sample weighting, fairness
  constraints).</span>

## Liên kết concepts liên quan - <span class="en">Related Concepts</span>

[[crisp-dm]] · [[point-biserial-correlation]] · [[network-graph-categorical-eda]]

## Câu hỏi tự kiểm tra nhanh - <span class="en">Quick Self-Check</span>

1. Vì sao không thể dùng tương quan Pearson chuẩn giữa `age` và cột `income` (dạng chữ
   `<=50K`/`>50K`) mà không xử lý gì trước?
   <br><span class="en">Why can't you run ordinary Pearson correlation between `age` and the
   `income` column (text `<=50K`/`>50K`) without any preprocessing?</span>
2. `input_features.isnull().sum()` báo 0 cho cột `workclass` — điều này có nghĩa cột đó không
   thiếu dữ liệu nào không? Vì sao?
   <br><span class="en">`input_features.isnull().sum()` reports 0 for `workclass` — does that
   mean the column has no missing data at all? Why or why not?</span>
3. Cho ví dụ 1 tình huống dữ liệu thiếu mang tính **structural** (thiếu vì logic) khác với thiếu
   ngẫu nhiên.
   <br><span class="en">Give an example of **structural** missingness (missing by logic) as
   opposed to random missingness.</span>

Bộ câu hỏi đầy đủ hơn: [[wk1-quiz]].
<br><span class="en">Full question bank: [[wk1-quiz]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-notebook]] · [[wk1-practice]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[wk2-data-preparation-feature-engineering]] · [[overview]]
