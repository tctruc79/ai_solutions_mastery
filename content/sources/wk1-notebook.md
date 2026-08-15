---
type: source
title: "WK1 Notebook — Data Understanding (honeyg_ai_practitioner_wk1_Data_Understanding.ipynb)"
tags: [data-understanding, crisp-dm, eda, uci-adult]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai_practitioner_wk1_Data_Understanding.ipynb"
---

## Thông tin nguồn - <span class="en">Source Info</span>

File `honeyg_ai_practitioner_wk1_Data_Understanding.ipynb`, 98 cell (49 markdown + 49 code),
kernel `AIDS_Maryam_Miradi` (Python 3.13).
<br><span class="en">File `honeyg_ai_practitioner_wk1_Data_Understanding.ipynb`, 98 cells (49
markdown + 49 code), kernel `AIDS_Maryam_Miradi` (Python 3.13).</span>

Đây là notebook tuần 1 của [[maryam-miradi]], mở đầu chương trình AI Practitioner bằng giai đoạn
**Data Understanding** của CRISP-DM.
<br><span class="en">This is the Week-1 notebook of [[maryam-miradi]], opening the AI
Practitioner programme with the **Data Understanding** phase of CRISP-DM.</span>

## Mục tiêu buổi học - <span class="en">Learning Objectives</span>

Notebook tự khai báo 4 mục theo đúng CRISP-DM: Collect Initial Data → Describe Data → Explore
Data → Verify Data Quality.
<br><span class="en">The notebook declares 4 sections that map directly onto CRISP-DM: Collect
Initial Data → Describe Data → Explore Data → Verify Data Quality.</span>

## Cấu trúc notebook - <span class="en">Notebook Structure</span>

- **Setup and Installation** — cài `ucimlrepo`, `networkx`.
  <br><span class="en">**Setup and Installation** — install `ucimlrepo`, `networkx`.</span>
- **1. Collect Initial Data** — 1.1 Census Income Data, tải qua `fetch_ucirepo(id=2)`.
  <br><span class="en">**1. Collect Initial Data** — 1.1 Census Income Data, loaded via
  `fetch_ucirepo(id=2)`.</span>
- **2. Describe Data** — 2.1 Data Structure (shape, `.head()`), 2.2 Metadata (`adult.metadata`,
  file `.names` gốc từ UCI).
  <br><span class="en">**2. Describe Data** — 2.1 Data Structure (shape, `.head()`), 2.2 Metadata
  (`adult.metadata`, the original UCI `.names` file).</span>
- **3. Explore Data** — 3.1 General Exploration (`.info()`, `.describe()`, tỷ lệ % từng danh
  mục), 3.2 Correlations (heatmap Pearson cho biến số + Point-Biserial cho biến số ↔ nhị phân),
  3.3 Advanced Exploration (network graph education↔occupation).
  <br><span class="en">**3. Explore Data** — 3.1 General Exploration (`.info()`, `.describe()`,
  category percentage breakdowns), 3.2 Correlations (Pearson heatmap for numeric features +
  Point-Biserial for numeric↔binary), 3.3 Advanced Exploration (education↔occupation network
  graph).</span>
- **4. Verify Data Quality** — 4.1 Clean Data (chuẩn hóa nhãn `income`), 4.2 Impute Missing
  Values (nguyên tắc, chưa thực thi), 4.3 Ideas for Feature Generation / Observations (ghi chú
  fairness).
  <br><span class="en">**4. Verify Data Quality** — 4.1 Clean Data (normalise the `income`
  label), 4.2 Impute Missing Values (principle only, not yet executed), 4.3 Ideas for Feature
  Generation / Observations (fairness notes).</span>

## Thư viện & công cụ - <span class="en">Libraries & Tools</span>

`sys`, `packaging.version`, `requests`, `pprint`, `json` (tiện ích chung); `numpy`, `pandas`
(dữ liệu); `seaborn`, `matplotlib` (trực quan hóa); `scipy.stats.pointbiserialr` (thống kê);
`networkx` (phân tích mạng); `ucimlrepo.fetch_ucirepo` (tải dữ liệu); `sklearn` (chỉ dùng để
kiểm tra phiên bản, chưa build model tuần này).
<br><span class="en">`sys`, `packaging.version`, `requests`, `pprint`, `json` (general
utilities); `numpy`, `pandas` (data); `seaborn`, `matplotlib` (visualisation);
`scipy.stats.pointbiserialr` (statistics); `networkx` (network analysis);
`ucimlrepo.fetch_ucirepo` (data loading); `sklearn` (version check only, no model built this
week).</span>

## Walkthrough kỹ thuật theo từng phần - <span class="en">Technical Walkthrough</span>

### 1. Collect & Describe

`fetch_ucirepo(id=2)` trả về `adult.data.features` (48,842 dòng × 14 cột) và
`adult.data.targets` (48,842 × 1, cột `income`). Metadata gốc từ UCI (`adult.metadata`) và file
mô tả `.names` được tải bằng hàm tự viết `download_file()` để đối chiếu.
<br><span class="en">`fetch_ucirepo(id=2)` returns `adult.data.features` (48,842 rows × 14
columns) and `adult.data.targets` (48,842 × 1, column `income`). The original UCI metadata
(`adult.metadata`) and the `.names` description file are fetched with a hand-written
`download_file()` helper for cross-checking.</span>

### 2. Explore — thống kê mô tả và mất cân bằng dữ liệu

`input_features.isnull().sum()` cho thấy `workclass` (963 dòng) và `occupation` (966 dòng) có
giá trị trống — thực ra là ký tự `'?'` chứ không phải `NaN` chuẩn. Cột `income` ban đầu có **4**
nhãn tưởng như khác nhau: `<=50K`, `<=50K.`, `>50K`, `>50K.` — do trộn lẫn phần train (không có
dấu chấm) và test (có dấu chấm) của UCI. Phân phối giới tính lệch mạnh: Nam 66.85% / Nữ 33.15%;
chủng tộc lệch về `White` 85.50%.
<br><span class="en">`input_features.isnull().sum()` shows `workclass` (963 rows) and
`occupation` (966 rows) with blanks — actually the character `'?'`, not standard `NaN`. The
`income` column initially has **4** seemingly distinct labels: `<=50K`, `<=50K.`, `>50K`,
`>50K.` — a mix of UCI's train portion (no trailing dot) and test portion (trailing dot). Gender
is heavily skewed: Male 66.85% / Female 33.15%; race skews toward `White` at 85.50%.</span>

### 3. Correlation — 2 kỹ thuật khác nhau cho 2 kiểu biến

Với các cột số ↔ số: heatmap tương quan Pearson chuẩn (`.corr()` + `sns.heatmap`). Với cột số ↔
target nhị phân (`income`): **tương quan điểm-nhị phân (point-biserial)** qua
`scipy.stats.pointbiserialr` — xem [[point-biserial-correlation]]. Kết quả:
`education-num` (0.26) và `age` (0.18) tương quan dương mạnh nhất với thu nhập cao.
<br><span class="en">For numeric ↔ numeric columns: a standard Pearson correlation heatmap
(`.corr()` + `sns.heatmap`). For numeric ↔ binary target (`income`): **point-biserial
correlation** via `scipy.stats.pointbiserialr` — see [[point-biserial-correlation]]. Result:
`education-num` (0.26) and `age` (0.18) correlate most strongly (positively) with high
income.</span>

### 3.3 Network Analysis — trực quan hóa quan hệ phân loại

Dùng `networkx.DiGraph` để vẽ 10 cặp `education → occupation` phổ biến nhất (đếm bằng
`groupby(['education','occupation']).size()`), xem [[network-graph-categorical-eda]].
<br><span class="en">Uses `networkx.DiGraph` to draw the top-10 most common `education →
occupation` pairs (counted via `groupby(['education','occupation']).size()`), see
[[network-graph-categorical-eda]].</span>

### 4. Verify Data Quality

`target_feature['income'].str.replace('.', '', regex=False).str.strip()` gộp 4 nhãn về đúng 2:
`<=50K` (37,155) / `>50K` (11,687). Notebook nhấn mạnh nguyên tắc: **không tự động điền giá trị
thiếu bằng 0/trung bình mà không điều tra trước** — mục 4.2 chỉ nêu nguyên tắc, chưa impute thật.
<br><span class="en">`target_feature['income'].str.replace('.', '', regex=False).str.strip()`
collapses the 4 labels into exactly 2: `<=50K` (37,155) / `>50K` (11,687). The notebook
emphasises a principle: **never auto-fill missing values with 0/mean without investigating
first** — section 4.2 states the principle only, no actual imputation is performed yet.</span>

## Kết quả/số liệu nổi bật - <span class="en">Key Numbers</span>

| Chỉ số - <span class="en">Metric</span> | Giá trị - <span class="en">Value</span> |
|---|---|
| Số dòng / cột gốc | 48,842 × 14 (input) + 1 (target) |
| `income` sau khi làm sạch | `<=50K` 37,155 (76.1%) · `>50K` 11,687 (23.9%) |
| `workclass` thiếu (`'?'`) | 1,836 (3.76%) |
| `occupation` thiếu (`'?'`) | 1,843 (~3.8%, sau gộp) |
| Tương quan điểm-nhị phân mạnh nhất với `income` | `education-num` r≈0.26, `age` r≈0.18 |
| `capital-gain` > 0 | 4,035 / 48,842 dòng (8.3%) |

## Vấn đề/quirk phát hiện trong code gốc - <span class="en">Quirks / Issues Found</span>

Notebook load `input_features`/`target_feature` **riêng biệt** nhiều lần qua các cell khác nhau
(cell 68 làm sạch `all_features['income']` cục bộ, cell 93 làm sạch lại `target_feature` — hai
thao tác tương tự nhưng không dùng chung 1 hàm) — đây chính là lý do tuần 2 tách hẳn logic này
thành hàm `clean_y()` tái sử dụng được, xem [[wk2-notebook]].
<br><span class="en">The notebook loads `input_features`/`target_feature` **separately** across
several cells (cell 68 cleans `all_features['income']` locally, cell 93 re-cleans
`target_feature` — two similar operations not sharing one function) — this is exactly why Week 2
extracts this logic into a reusable `clean_y()` function, see [[wk2-notebook]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-data-understanding]] · [[wk1-practice]] · [[crisp-dm]] · [[point-biserial-correlation]] ·
[[network-graph-categorical-eda]] · [[wk2-notebook]] · [[overview]]
