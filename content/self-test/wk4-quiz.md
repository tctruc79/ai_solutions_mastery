---
type: self-test
title: "WK4 Self-Test — Model Selection, Explainability, Fairness, Deployment"
tags: [self-test, wk4, shap, fairlearn, pca]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

Phạm vi: [[wk4-model-selection-explainability-fairness-deployment]], [[wk4-notebook]]. Câu hỏi
tự luyện Claude tự soạn, không phải đề thi chính thức.
<br><span class="en">Scope: [[wk4-model-selection-explainability-fairness-deployment]],
[[wk4-notebook]]. Practice questions authored by Claude, not an official exam.</span>

## Câu 1 - <span class="en">Q1</span>

XGBoost + CTGAN thắng lưới 54 tổ hợp với AUPRC 0.9639 trên cross-validation, nhưng chỉ đạt 0.7951
trên test set thật — THẤP HƠN cả XGBoost không resample (0.8024). Giải thích nguyên nhân.
<br><span class="en">XGBoost + CTGAN wins the 54-combination grid with a 0.9639
cross-validation AUPRC, but only reaches 0.7951 on the real test set — LOWER than non-resampled
XGBoost (0.8024). Explain why.</span>

**Đáp án - <span class="en">Answer</span>**: Mẫu tổng hợp (SMOTE/CTGAN) thường "gần" và "dễ đoán"
hơn dữ liệu thật vì được sinh dựa trên chính các điểm dữ liệu đó — cross-validation đo trên tập
đã LẪN các mẫu dễ đoán này cho điểm cao giả tạo. Test set thật (chưa từng resample) mới phản ánh
đúng hiệu năng khi gặp dữ liệu hoàn toàn mới. Xem
[[wk4-model-selection-explainability-fairness-deployment]] mục 1.
<br><span class="en">Synthetic samples (SMOTE/CTGAN) tend to be "closer to" and "easier to
predict" than real data since they're generated from those very points — cross-validation
measured on a set mixed with these easy synthetic samples produces an artificially inflated
score. The real test set (never resampled) is what actually reflects performance on genuinely new
data. See [[wk4-model-selection-explainability-fairness-deployment]] section 1.</span>

## Câu 2 - <span class="en">Q2</span>

Notebook gốc in ra "Best model based on AUPRC: Extra Trees SMOTE with an AUPRC of 0.9026" nhưng
bảng sắp xếp đúng ngay bên dưới cho thấy người thắng thật là XGBoost + CTGAN (0.9639). Lỗi kỹ
thuật chính xác nằm ở đâu?
<br><span class="en">The original notebook prints "Best model based on AUPRC: Extra Trees SMOTE
with an AUPRC of 0.9026" but the correctly sorted table right below shows the real winner is
XGBoost + CTGAN (0.9639). Where exactly is the technical bug?</span>

**Đáp án - <span class="en">Answer</span>**: `max(results, key=lambda x: x[1])` lặp qua CÁC KHÓA
(chuỗi tên model) của dict, không phải cặp (khóa, giá trị) — `x[1]` do đó là ký tự thứ 2 của tên
model, không phải điểm AUPRC. Cách sửa đúng: `max(results.items(), key=lambda kv: kv[1][0])`. Xem
[[wk4-notebook]] mục quirk #1.
<br><span class="en">`max(results, key=lambda x: x[1])` iterates over the dict's KEYS (model-name
strings), not (key, value) pairs — `x[1]` is therefore the model name's 2nd character, not an
AUPRC score. Correct fix: `max(results.items(), key=lambda kv: kv[1][0])`. See [[wk4-notebook]]
quirk #1.</span>

## Câu 3 - <span class="en">Q3</span>

PCA giữ 95% phương sai cần 10/19 thành phần. Điều này nói lên điều gì về 19 đặc trưng gốc?
<br><span class="en">PCA needs 10/19 components to retain 95% of variance. What does this say
about the original 19 features?</span>

**Đáp án - <span class="en">Answer</span>**: 19 đặc trưng gốc có khá nhiều REDUNDANCY (thông tin
trùng lặp lẫn nhau) — không cần giữ nguyên cả 19 chiều mới nắm được phần lớn thông tin, hơn một
nửa số chiều có thể "nén" lại mà không mất nhiều. Xem [[pca-tsne-dimensionality-reduction]].
<br><span class="en">The 19 original features have considerable REDUNDANCY (overlapping
information) — you don't need all 19 dimensions to capture most of the information, over half the
dimensions can be "compressed away" without losing much. See
[[pca-tsne-dimensionality-reduction]].</span>

## Câu 4 - <span class="en">Q4</span>

K-Means cần biết trước K, DBSCAN thì không. Vậy DBSCAN "biết" số cụm bằng cách nào, và nó có khái
niệm gì mà K-Means không có?
<br><span class="en">K-Means needs K known in advance, DBSCAN doesn't. So how does DBSCAN "know"
the cluster count, and what concept does it have that K-Means lacks?</span>

**Đáp án - <span class="en">Answer</span>**: DBSCAN tự xác định số cụm dựa trên MẬT ĐỘ điểm
(`eps`, `min_samples`) — cụm hình thành từ các vùng đủ dày đặc. Nó có khái niệm NHIỄU (nhãn -1)
cho điểm không thuộc cụm nào, trong khi K-Means BẮT BUỘC gán mọi điểm vào 1 cụm nào đó. Xem
[[kmeans-dbscan-clustering]].
<br><span class="en">DBSCAN determines the cluster count itself based on point DENSITY (`eps`,
`min_samples`) — clusters form from sufficiently dense regions. It has a NOISE concept (label -1)
for points belonging to no cluster, while K-Means MUST assign every point to some cluster. See
[[kmeans-dbscan-clustering]].</span>

## Câu 5 - <span class="en">Q5</span>

SHAP "global" (`summary_plot`) và SHAP "local" (`waterfall`) trả lời 2 câu hỏi khác nhau như thế
nào? Cho ví dụ tình huống cần dùng local thay vì global.
<br><span class="en">How do SHAP "global" (`summary_plot`) and SHAP "local" (`waterfall`) answer
different questions? Give an example situation needing local instead of global.</span>

**Đáp án - <span class="en">Answer</span>**: Global trả lời "đặc trưng nào quan trọng nhất NÓI
CHUNG trên toàn dữ liệu?"; local trả lời "vì sao model dự đoán ĐÚNG DÒNG NÀY như vậy?". Cần local
khi phải giải trình 1 quyết định cụ thể — vd giải thích cho 1 khách hàng vì sao đơn vay của họ bị
từ chối, global không trả lời được câu hỏi ở mức cá nhân đó. Xem [[shap]].
<br><span class="en">Global answers "which feature matters most OVERALL across the whole
dataset?"; local answers "why did the model predict THIS SPECIFIC row the way it did?". Local is
needed when a specific decision must be justified — e.g. explaining to one customer why their
loan application was rejected, global can't answer that individual-level question. See
[[shap]].</span>

## Câu 6 - <span class="en">Q6</span>

FairLearn có 3 cách can thiệp: in-processing, post-processing, và điều chỉnh thủ công. Khác nhau
ở điểm nào trong pipeline, và cách nào KHÔNG cần huấn luyện lại model?
<br><span class="en">FairLearn offers 3 intervention approaches: in-processing, post-processing,
and manual adjustment. Where do they differ in the pipeline, and which one does NOT require
retraining the model?</span>

**Đáp án - <span class="en">Answer</span>**: In-processing (`ExponentiatedGradient`) đưa ràng buộc
công bằng vào NGAY lúc huấn luyện — cần huấn luyện lại. Post-processing (`ThresholdOptimizer`)
chỉ chỉnh NGƯỠNG phân loại sau khi đã có model — KHÔNG cần huấn luyện lại. Điều chỉnh thủ công
cũng không cần huấn luyện lại nhưng thiếu đảm bảo toán học của 2 cách trên. Xem [[fairlearn]].
<br><span class="en">In-processing (`ExponentiatedGradient`) bakes the fairness constraint into
training itself — requires retraining. Post-processing (`ThresholdOptimizer`) only adjusts the
classification THRESHOLD after the model already exists — does NOT require retraining. Manual
adjustment also doesn't require retraining but lacks the mathematical guarantees of the two
approaches above. See [[fairlearn]].</span>

## Câu 7 - <span class="en">Q7</span>

Model cuối cùng được `joblib.dump` để triển khai đã được Hyperopt tinh chỉnh trên dữ liệu CTGAN
(CV AUPRC 0.9653) — nhưng notebook có xác nhận lại AUPRC của CHÍNH bản đã tinh chỉnh này trên
test set thật không? Vì sao đây là 1 khoảng trống đáng chú ý?
<br><span class="en">The final model `joblib.dump`ed for deployment was Hyperopt-tuned on CTGAN
data (0.9653 CV AUPRC) — did the notebook re-confirm THIS tuned model's AUPRC on the real test
set? Why is this a notable gap?</span>

**Đáp án - <span class="en">Answer</span>**: Không — mục "Testing on Test Set" chỉ kiểm tra 2
model CHƯA tinh chỉnh (gốc: 0.8024, CTGAN: 0.7951); model đã tinh chỉnh không bao giờ được đo lại
trên test set thật trong notebook. Đây là khoảng trống đáng chú ý vì chính mục ngay phía trên đã
chứng minh CV cao không đảm bảo test cao — bỏ qua bước xác nhận cuối cùng này có nghĩa hiệu năng
thật của model TRIỂN KHAI vẫn chưa được biết chắc chắn. Xem [[wk4-notebook]] mục quirk #2.
<br><span class="en">No — the "Testing on Test Set" section only checks 2 UNTUNED models
(original: 0.8024, CTGAN: 0.7951); the tuned model is never re-measured on the real test set in
the notebook. This is a notable gap because the section right above it just proved high CV
doesn't guarantee high test performance — skipping this final confirmation means the DEPLOYED
model's true performance remains unknown with certainty. See [[wk4-notebook]] quirk #2.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-model-selection-explainability-fairness-deployment]] · [[wk4-notebook]] · [[wk3-quiz]] ·
[[cumulative-quiz]]
