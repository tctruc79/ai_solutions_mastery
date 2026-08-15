---
type: self-test
title: "WK3 Self-Test — Modelling II: 9 Models, XGBoost, Imbalance, Deep Learning"
tags: [self-test, wk3, xgboost, imbalanced-data]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Phạm vi: [[wk3-modelling-imbalance-deep-learning]], [[wk3-notebook]]. Câu hỏi tự luyện Claude tự
soạn, không phải đề thi chính thức.
<br><span class="en">Scope: [[wk3-modelling-imbalance-deep-learning]], [[wk3-notebook]].
Practice questions authored by Claude, not an official exam.</span>

## Câu 1 - <span class="en">Q1</span>

Vì sao AUPRC được chọn làm thước đo chính thức từ tuần này, thay vì ROC-AUC (vốn cũng "chịu được"
mất cân bằng tốt hơn accuracy)?
<br><span class="en">Why is AUPRC chosen as the official metric from this week on, instead of
ROC-AUC (which is also considered more robust to imbalance than accuracy)?</span>

**Đáp án - <span class="en">Answer</span>**: ROC-AUC tính cả True Negative Rate — với lớp âm
chiếm 76% dữ liệu, TNR gần như luôn cao dù mô hình dở ở lớp thiểu số, khiến ROC-AUC vẫn có thể
"trông tốt" một cách ảo. AUPRC chỉ dùng Precision và Recall, cả 2 đều tính TRÊN lớp dương (thiểu
số), nên phản ánh trung thực hơn hiệu năng thực trên lớp cần quan tâm. Xem [[auprc]].
<br><span class="en">ROC-AUC factors in the True Negative Rate — with the negative class at 76%
of the data, TNR stays high almost regardless of how poor the model is on the minority class, so
ROC-AUC can still look deceptively good. AUPRC only uses Precision and Recall, both computed ON
the positive (minority) class, so it more honestly reflects real performance on the class that
matters. See [[auprc]].</span>

## Câu 2 - <span class="en">Q2</span>

`scale_pos_weight` trong XGBoost và việc dùng SMOTE trước khi huấn luyện đều nhằm "xử lý mất cân
bằng" — chúng khác nhau ở đâu trong quy trình?
<br><span class="en">`scale_pos_weight` in XGBoost and applying SMOTE before training both
"handle imbalance" — where do they actually differ in the workflow?</span>

**Đáp án - <span class="en">Answer</span>**: `scale_pos_weight` điều chỉnh TRỌNG SỐ HỌC ngay
trong quá trình huấn luyện — dữ liệu đầu vào không đổi kích thước, chỉ có mô hình "chú ý" hơn tới
lớp thiểu số khi tính loss. SMOTE thay đổi CHÍNH DỮ LIỆU trước khi mô hình thấy nó — sinh thêm mẫu
thiểu số để tỷ lệ lớp cân bằng ngay từ đầu vào. Có thể dùng độc lập hoặc kết hợp cả hai. Xem
[[xgboost]] và [[imbalanced-data-resampling]].
<br><span class="en">`scale_pos_weight` adjusts the LEARNING WEIGHT during training itself — the
input data doesn't change size, the model just "pays more attention" to the minority class when
computing loss. SMOTE changes the DATA ITSELF before the model ever sees it — generating extra
minority samples so the class ratio is balanced from the input onward. Both can be used
independently or combined. See [[xgboost]] and [[imbalanced-data-resampling]].</span>

## Câu 3 - <span class="en">Q3</span>

SMOTE, ADASYN và Random Undersampling đều "cân bằng lớp" nhưng bằng cơ chế khác nhau. Mô tả ngắn
gọn từng cơ chế.
<br><span class="en">SMOTE, ADASYN, and Random Undersampling all "balance classes" but through
different mechanisms. Briefly describe each mechanism.</span>

**Đáp án - <span class="en">Answer</span>**: SMOTE — nội suy tuyến tính giữa 1 mẫu thiểu số và 1
láng giềng gần nhất cùng lớp để tạo mẫu mới. ADASYN — như SMOTE nhưng sinh nhiều mẫu hơn ở vùng
biên (mẫu thiểu số có nhiều láng giềng đa số bao quanh). Random Undersampling — bớt ngẫu nhiên
mẫu đa số cho bằng số lượng thiểu số, không sinh gì mới. Xem [[imbalanced-data-resampling]].
<br><span class="en">SMOTE — linearly interpolates between a minority sample and its nearest
same-class neighbour to create a new sample. ADASYN — like SMOTE but generates more samples in
border regions (minority samples surrounded by many majority neighbours). Random Undersampling —
randomly drops majority samples to match the minority count, generates nothing new. See
[[imbalanced-data-resampling]].</span>

## Câu 4 - <span class="en">Q4</span>

Mô hình Deep Learning đầu tiên trong notebook cho `loss: nan` suốt cả 20 epoch. Nguyên nhân kỹ
thuật chính xác là gì, và cách sửa?
<br><span class="en">The first Deep Learning model in the notebook produces `loss: nan`
throughout all 20 epochs. What's the exact technical cause, and the fix?</span>

**Đáp án - <span class="en">Answer</span>**: Lớp cuối dùng `activation='tanh'` (miền giá trị
[-1, 1]) nhưng hàm mất mát là `categorical_crossentropy` (yêu cầu xác suất trong [0, 1]) — kết
hợp sai này gây `nan` ngay từ vòng lặp đầu. Sửa bằng cách đổi activation lớp cuối sang `sigmoid`
(nhị phân, 1 output) hoặc `softmax` (đa lớp) để khớp với cross-entropy loss. Xem
[[wk3-modelling-imbalance-deep-learning]] mục 6.
<br><span class="en">The final layer uses `activation='tanh'` (range [-1, 1]) but the loss is
`categorical_crossentropy` (requires probabilities in [0, 1]) — this wrong pairing causes `nan`
from the very first round. Fix by changing the final activation to `sigmoid` (binary, 1 output)
or `softmax` (multi-class) to match cross-entropy loss. See
[[wk3-modelling-imbalance-deep-learning]] section 6.</span>

## Câu 5 - <span class="en">Q5</span>

Notebook gốc chạy `smote.fit_resample()` trên TOÀN BỘ tập train MỘT LẦN, rồi mới `cross_val_score`
qua 10-fold trên dữ liệu đã resample đó. Vì sao cách làm này tiềm ẩn rủi ro, và cách làm đúng hơn
là gì?
<br><span class="en">The original notebook runs `smote.fit_resample()` on the ENTIRE training set
ONCE, then `cross_val_score` over 10 folds on that already-resampled data. Why does this carry a
risk, and what's the more correct approach?</span>

**Đáp án - <span class="en">Answer</span>**: 1 mẫu tổng hợp SMOTE trong fold validation có thể
được nội suy từ 1 mẫu thật đang nằm trong fold train (vì resample xảy ra TRƯỚC khi chia fold) —
khiến validation fold không còn hoàn toàn "chưa từng thấy", có thể làm điểm AUPRC lạc quan giả.
Cách đúng hơn: đặt SMOTE bên trong 1 `imblearn.pipeline.Pipeline`, để resampling chỉ dùng dữ liệu
của phần train trong MỖI fold riêng biệt. Xem [[wk3-modelling-imbalance-deep-learning]] mục "Lỗi
thường gặp".
<br><span class="en">A SMOTE synthetic sample in the validation fold may have been interpolated
from a real sample sitting in the train fold (since resampling happens BEFORE the fold split) —
so the validation fold is no longer fully "unseen", potentially inflating the AUPRC score. The
more correct approach: put SMOTE inside an `imblearn.pipeline.Pipeline`, so resampling only uses
that fold's own training portion. See [[wk3-modelling-imbalance-deep-learning]], "Pitfalls"
section.</span>

## Câu 6 - <span class="en">Q6</span>

XGBoost dùng `importance_type='gain'` để tính feature importance thay vì `'weight'`. Khác nhau ở
điểm nào, và vì sao `gain` phù hợp hơn?
<br><span class="en">XGBoost uses `importance_type='gain'` to compute feature importance instead
of `'weight'`. What's the difference, and why is `gain` more appropriate?</span>

**Đáp án - <span class="en">Answer</span>**: `weight` đếm SỐ LẦN 1 đặc trưng được dùng để chia
nhánh trong toàn bộ các cây — chỉ phản ánh TẦN SUẤT xuất hiện. `gain` đo mức GIẢM LOSS THỰC TẾ mà
đặc trưng đó đóng góp mỗi lần được dùng — phản ánh đúng MỨC ĐỘ QUAN TRỌNG hơn, vì 1 đặc trưng có
thể được dùng ít lần nhưng mỗi lần đóng góp rất lớn vào việc giảm loss. Xem [[xgboost]].
<br><span class="en">`weight` counts the NUMBER OF TIMES a feature is used to split across all
trees — only reflects FREQUENCY. `gain` measures the ACTUAL LOSS REDUCTION that feature
contributes each time it's used — more accurately reflecting IMPORTANCE, since a feature might be
used rarely but contribute a large loss reduction each time. See [[xgboost]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-modelling-imbalance-deep-learning]] · [[wk3-notebook]] · [[wk2-quiz]] · [[cumulative-quiz]]
