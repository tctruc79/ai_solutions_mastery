---
type: assignment
title: "WK1 Assignment — Data Understanding (Bank Marketing Dataset)"
tags: [assignment, data-understanding, bank-marketing, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Assignments/honeyg_Assignment_wk1_Data_Understanding.ipynb"
also_covers:
  - "raw/Assignments/Tran Chanh Truc_wk1_assessment_feedback_Dr_Maryam_Miradi.pdf"
---

## Đề bài - <span class="en">Assignment Brief</span>

Áp dụng đúng phương pháp Data Understanding của [[wk1-data-understanding]] (Collect → Describe
→ Explore → Verify Data Quality theo [[crisp-dm]]) lên **một bộ dữ liệu khác** do học viên tự
chọn — không lặp lại UCI Adult của bài giảng, mục đích kiểm tra khả năng tổng quát hóa quy trình
sang bài toán mới.
<br><span class="en">Apply the exact Data Understanding methodology from
[[wk1-data-understanding]] (Collect → Describe → Explore → Verify Data Quality per
[[crisp-dm]]) to a **different dataset** chosen by the student — not a repeat of the lecture's
UCI Adult dataset, testing the ability to generalise the process to a new problem.</span>

Trực chọn **UCI Bank Marketing Dataset** (id=222) — dự đoán khách hàng có đăng ký gửi tiết kiệm
kỳ hạn (`y` = yes/no) hay không, dựa trên dữ liệu chiến dịch tiếp thị qua điện thoại của 1 ngân
hàng Bồ Đào Nha (5/2008–11/2010).
<br><span class="en">Trực chose the **UCI Bank Marketing Dataset** (id=222) — predicting whether
a client subscribes to a term deposit (`y` = yes/no) based on phone-marketing campaign data from
a Portuguese bank (May 2008 – Nov 2010).</span>

## Trực đã làm gì - <span class="en">What Was Done</span>

Notebook 448 cell — quy mô lớn hơn nhiều so với bản lecture gốc (98 cell) vì bộ Bank Marketing
đóng gói phức tạp hơn (zip lồng zip, 4 biến thể file: `bank.csv`/`bank-full.csv`/
`bank-additional.csv`/`bank-additional-full.csv`) và Trực mở rộng thêm nhiều phân tích ngoài yêu
cầu tối thiểu.
<br><span class="en">A 448-cell notebook — much larger than the original lecture (98 cells)
because the Bank Marketing archive is more complex (zip nested inside zip, 4 file variants:
`bank.csv`/`bank-full.csv`/`bank-additional.csv`/`bank-additional-full.csv`) and Trực extended
the analysis well beyond the minimum requirement.</span>

Điểm nổi bật so với bản lecture gốc:
<br><span class="en">Highlights beyond the original lecture:</span>

- **Tải dữ liệu song song 2 nguồn** — vừa viết `download_file()` tùy biến để giải nén zip lồng
  zip, vừa dùng `fetch_ucirepo(id=222)`, rồi đối chiếu chéo để xác nhận cả 2 khớp nhau.
  <br><span class="en">**Dual-source loading** — a customised `download_file()` handling the
  nested-zip extraction, alongside `fetch_ucirepo(id=222)`, cross-checked against each other for
  consistency.</span>
- **Thư viện hàm tái sử dụng** — `categorical_summary()`, `missing_value_summary()`,
  `plot_numeric_distributions()`, `group_numeric_summary_flat()`,
  `group_categorical_summary_flat()`: mỗi hàm 1 nhiệm vụ, có docstring, dùng lặp lại xuyên suốt
  notebook thay vì copy-paste code.
  <br><span class="en">**Reusable function library** — `categorical_summary()`,
  `missing_value_summary()`, `plot_numeric_distributions()`, `group_numeric_summary_flat()`,
  `group_categorical_summary_flat()`: single-responsibility, docstring'd, reused throughout
  instead of copy-pasted.</span>
- **Phát hiện rò rỉ dữ liệu (data leakage)** — chủ động gắn cờ biến `duration` (thời lượng cuộc
  gọi) là biến "hậu-sự-kiện": chỉ biết được SAU khi cuộc gọi kết thúc, nên không được dùng để dự
  đoán TRƯỚC khi gọi — một quan sát về ML-awareness mà đề bài không yêu cầu tường minh.
  <br><span class="en">**Data-leakage detection** — proactively flagging `duration` (call
  length) as a post-hoc variable: only known AFTER the call ends, so it cannot be used to predict
  BEFORE calling — an ML-awareness insight the brief did not explicitly require.</span>
- **63 violin plot phân tích 2 biến (bivariate)** cho 7 biến số × 9 biến phân loại, và **5 cụm
  network graph** (nhân khẩu học, hành vi tài chính, kênh marketing, cường độ chiến dịch, mạng
  lõi với target) — mở rộng kỹ thuật [[network-graph-categorical-eda]] xa hơn 1 cặp biến duy
  nhất của bản lecture gốc.
  <br><span class="en">**63 bivariate violin plots** across 7 numeric × 9 categorical features,
  and **5 network-graph clusters** (demographics, financial behaviour, marketing channel,
  campaign intensity, core target network) — extending [[network-graph-categorical-eda]] well
  beyond the single variable pair in the original lecture.</span>
- **Bảng tổng hợp chất lượng dữ liệu (mục 4.6)** — 8 vấn đề chất lượng liệt kê có hệ thống: mức
  độ nghiêm trọng, dữ liệu bị ảnh hưởng, giải pháp đề xuất (vd `duration` rò rỉ → loại khỏi mô
  hình dự đoán trước cuộc gọi; `poutcome` thiếu 81.7% → giữ làm 1 category riêng thay vì impute).
  <br><span class="en">**Data-quality summary table (section 4.6)** — 8 issues listed
  systematically: severity, affected data, proposed fix (e.g. leaking `duration` → excluded from
  the pre-call prediction model; `poutcome` missing 81.7% → kept as its own category rather than
  imputed).</span>

## Đối chiếu với bài giảng gốc - <span class="en">Comparison with the Lecture</span>

Đúng quy trình 4 bước [[crisp-dm]] và tái sử dụng 2 kỹ thuật cốt lõi của tuần 1 nguyên xi:
[[point-biserial-correlation]] (mục 3.2.4.2, so `duration`/`pdays`/`previous`/`campaign` với
target `y`) và [[network-graph-categorical-eda]] (mở rộng thành 5 cụm thay vì 1 cặp). Khác biệt
lớn nhất: bản lecture chỉ có 1 nguồn dữ liệu (UCI Adult), còn bài này phải xử lý 4 file với 2
schema khác nhau (family `bank`/`bank-full` 16 cột vs family `bank-additional`/
`bank-additional-full` 20 cột, thêm 5 chỉ số kinh tế vĩ mô) — đòi hỏi thêm bước xác minh tính
nhất quán schema (mục 4.5) mà bản lecture gốc không cần vì chỉ có 1 nguồn duy nhất.
<br><span class="en">Follows the same [[crisp-dm]] 4-step process and reuses two Week-1 core
techniques as-is: [[point-biserial-correlation]] (section 3.2.4.2, comparing `duration`/
`pdays`/`previous`/`campaign` against target `y`) and [[network-graph-categorical-eda]]
(extended to 5 clusters instead of one pair). The biggest difference: the lecture has a single
data source (UCI Adult), while this assignment must handle 4 files across 2 different schemas
(the `bank`/`bank-full` family, 16 columns, vs the `bank-additional`/`bank-additional-full`
family, 20 columns, adding 5 macroeconomic indicators) — requiring an extra schema-consistency
verification step (section 4.5) that the original lecture didn't need with only one
source.</span>

## Feedback từ Dr. Maryam Miradi - <span class="en">Feedback from Dr. Maryam Miradi</span>

**Điểm: 92/100 — Strong pass, A-.**
<br><span class="en">**Score: 92/100 — Strong pass, A-.**</span>

| Tiêu chí - <span class="en">Criterion</span> | Điểm - <span class="en">Score</span> |
|---|---|
| Data collection & loading | 95% |
| Data description & metadata | 92% |
| Exploratory analysis depth | 90% |
| Visualisation quality | 88% |
| Correlation analysis | 93% |
| Data quality verification | 82% |
| Findings & insights report | 95% |
| Code quality & reusability | 85% |

Nhận xét tổng: "Excellent depth and reusable function architecture throughout... Two bugs in
section 4 prevent a top band score but do not diminish the quality of the analytical work."
<br><span class="en">Overall comment: "Excellent depth and reusable function architecture
throughout... Two bugs in section 4 prevent a top band score but do not diminish the quality of
the analytical work."</span>

**2 lỗi được nêu:**
<br><span class="en">**2 bugs flagged:**</span>

1. Trong bước đổi tên cột `day` → `day_of_week`, 3 trong 4 lệnh rename đầu tiên bị chuyền nhầm
   biến — cả 3 đều nối tiếp từ `bank_full_input_features` thay vì từ chính biến của chúng, âm
   thầm GHI ĐÈ dữ liệu của `bank_input_features`, `bank_additional_full_input_features`,
   `bank_additional_input_features` bằng dữ liệu của `bank-full.csv` — nguy hiểm hơn vì
   `bank-additional-full` là 1 schema HOÀN TOÀN KHÁC (thêm 5 biến kinh tế vĩ mô), nên lỗi này
   làm mất hẳn 1 bộ đặc trưng mà không báo lỗi gì.
   <br><span class="en">In the `day` → `day_of_week` rename step, 3 of the 4 initial rename
   calls were chained off the wrong variable — all three continued from
   `bank_full_input_features` instead of their own variable, silently OVERWRITING
   `bank_input_features`, `bank_additional_full_input_features`, and
   `bank_additional_input_features` with `bank-full.csv` data — more dangerous because
   `bank-additional-full` is an ENTIRELY DIFFERENT schema (adds 5 macroeconomic variables), so
   this bug silently loses a whole feature set with no error raised.</span>
2. Mục 4.3 "Feature generation / Observations" còn sót nguyên văn nội dung từ bài giảng UCI
   Adult (nhắc tới `workclass`, `Capital_Gain`, thống kê công bằng giới tính/chủng tộc, link US
   Census) — các trường này KHÔNG tồn tại trong bộ Bank Marketing, đoạn này cần viết lại cho
   đúng dataset.
   <br><span class="en">Section 4.3 "Feature generation / Observations" still contained
   copy-pasted content from the UCI Adult lecture (referencing `workclass`, `Capital_Gain`,
   gender/race fairness statistics, US Census links) — none of these fields exist in the Bank
   Marketing dataset; the section needed rewriting for the actual dataset.</span>

**Trạng thái hiện tại (đối chiếu 2026-08-14):** notebook trong `raw/Assignments/` ĐÃ khắc phục cả
2 lỗi trên. Mục rename giờ dùng đúng pattern lặp qua `datasets = [bank_full_input_features,
bank_input_features, bank_additional_full_input_features, bank_additional_input_features]` +
`for df in datasets: df.rename(...)` (đúng khuyến nghị DRY của feedback), kèm cell verify ngay
sau đó xác nhận cả 4 dataset giữ đúng shape gốc. Mục 4.3 gốc đã được thay bằng chuỗi mục 4.3–4.6
đúng nội dung Bank Marketing (Duplicate Records, Validity & Range Checks, Cross-Dataset
Structural Consistency, Data Quality Verification Summary).
<br><span class="en">**Current status (verified 2026-08-14):** the notebook in
`raw/Assignments/` HAS fixed both issues. The rename section now uses the correct loop pattern
over `datasets = [bank_full_input_features, bank_input_features,
bank_additional_full_input_features, bank_additional_input_features]` + `for df in datasets:
df.rename(...)` (matching the feedback's DRY recommendation), followed by a verification cell
confirming all 4 datasets retain their original shape. The original section 4.3 has been
replaced with a proper 4.3–4.6 sequence matching Bank Marketing content (Duplicate Records,
Validity & Range Checks, Cross-Dataset Structural Consistency, Data Quality Verification
Summary).</span>

Các khuyến nghị khác từ feedback cũng đã thấy dấu vết được áp dụng trong bản hiện tại: import
được gom hết vào 1 cell "Load Libraries" duy nhất (kể cả `sklearn`, `fetch_ucirepo` — trước đó bị
tách rời), và hàm `download_file()` có comment nội tuyến giải thích LÝ DO (vd vì sao kiểm tra
`.endswith('.zip')` 2 lần — do cấu trúc zip-lồng-zip của bộ dữ liệu này).
<br><span class="en">Other recommendations from the feedback also show signs of being applied
in the current version: imports are consolidated into a single "Load Libraries" cell (including
`sklearn`, `fetch_ucirepo` — previously split apart), and `download_file()` carries inline
comments explaining WHY (e.g. why `.endswith('.zip')` is checked twice — due to this dataset's
nested-zip structure).</span>

## Việc cần cải thiện cho tuần sau - <span class="en">Action Items for Next Week</span>

- Áp dụng thói quen **DRY qua vòng lặp** (đã học từ chính lỗi rename tuần này) cho các thao tác
  lặp lại tương tự ở tuần 2 (feature engineering trên nhiều cột cùng lúc) — kiểm tra ngay bằng
  `.info()`/`.shape` sau mỗi vòng lặp thay vì tin tưởng mù quáng.
  <br><span class="en">Apply the **DRY-via-loop** habit (learned directly from this week's
  rename bug) to similarly repeated operations in Week 2 (feature engineering across multiple
  columns at once) — verify immediately with `.info()`/`.shape` after each loop instead of
  blindly trusting it.</span>
- Tiếp tục vòng lặp "Analysis → Observation" ngay sau mỗi cell có output — bản hiện tại đã làm
  khá nhất quán ở phần 2–3, giữ vững thói quen này khi notebook wk2 phức tạp hơn (nhiều bước
  biến đổi dữ liệu nối tiếp nhau).
  <br><span class="en">Keep up the "Analysis → Observation" loop right after every
  output-producing cell — the current version does this fairly consistently in sections 2–3,
  keep the habit as the wk2 notebook grows more complex (many chained data-transformation
  steps).</span>
- Khi ingest wk2/wk3/wk4 assignment sau này, đối chiếu thêm xem các bài học rút ra ở đây (DRY
  loop, tách import, comment giải thích WHY) có được duy trì xuyên suốt không.
  <br><span class="en">When wk2/wk3/wk4 assignments are ingested later, also check whether the
  lessons learned here (DRY loops, consolidated imports, WHY-comments) are sustained
  throughout.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-data-understanding]] · [[wk1-notebook]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[maryam-miradi]]
