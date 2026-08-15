---
type: concept
title: "Phát Hiện Dị Thường: Isolation Forest, LOF, Autoencoder"
tags: [anomaly-detection, isolation-forest, lof, autoencoder, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**Phát hiện dị thường** (anomaly detection) là bài toán phát hiện các điểm dữ liệu KHÁC THƯỜNG so
với phần lớn dữ liệu còn lại, huấn luyện KHÔNG CẦN nhãn dị thường có sẵn. Notebook thử 3 kỹ
thuật: **Isolation Forest**, **Local Outlier Factor (LOF)**, và 1 **Autoencoder** tự viết bằng
Keras.
<br><span class="en">**Anomaly detection** is the task of finding data points that are UNUSUAL
relative to the bulk of the data, trained WITHOUT needing pre-existing anomaly labels. The
notebook tries 3 techniques: **Isolation Forest**, **Local Outlier Factor (LOF)**, and a
hand-written Keras **Autoencoder**.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Phân loại giám sát (XGBoost...) cần đủ ví dụ của CẢ HAI lớp để học ranh giới phân biệt. Nhưng
nhiều bài toán thực tế (gian lận thẻ, lỗi máy móc, xâm nhập mạng) có lớp "bất thường" CỰC HIẾM
hoặc THAY ĐỔI LIÊN TỤC (dạng mới chưa từng thấy) — không đủ ví dụ để huấn luyện phân loại thông
thường. Phát hiện dị thường học "thế nào là BÌNH THƯỜNG" từ lớp đa số, rồi đánh dấu bất kỳ điểm
nào khác biệt đủ nhiều — không cần biết trước hình dạng cụ thể của sự bất thường.
<br><span class="en">Supervised classification (XGBoost...) needs enough examples of BOTH classes
to learn a decision boundary. But many real problems (card fraud, equipment failure, network
intrusion) have an "abnormal" class that's EXTREMELY RARE or CONSTANTLY CHANGING (novel forms
never seen before) — not enough examples to train ordinary classification. Anomaly detection
learns "what NORMAL looks like" from the majority class, then flags anything sufficiently
different — without needing to know the specific shape of the anomaly in advance.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

**Isolation Forest**: xây nhiều cây quyết định NGẪU NHIÊN, mỗi cây tách dữ liệu bằng cách chọn
ngẫu nhiên 1 đặc trưng + 1 ngưỡng cắt — điểm DỊ THƯỜNG bị "cô lập" (tách ra 1 mình) SAU ÍT bước
cắt hơn điểm bình thường (vì nó khác biệt rõ), nên độ sâu trung bình cần để cô lập 1 điểm là điểm
số bất thường. **LOF**: so sánh MẬT ĐỘ láng giềng cục bộ của 1 điểm với mật độ láng giềng của
chính các láng giềng đó — điểm nằm ở vùng THƯA hơn hẳn xung quanh nó là dị thường. **Autoencoder**:
mạng neural học NÉN rồi TÁI TẠO lại chính đầu vào (encoder→bottleneck→decoder) — chỉ huấn luyện
trên dữ liệu bình thường, nên tái tạo TỐT dữ liệu bình thường nhưng tái tạo KÉM dữ liệu bất
thường (chưa từng thấy dạng đó) — sai số tái tạo (MSE) lớn = điểm khả nghi. Cả 3 đều huấn luyện
CHỈ trên lớp đa số (`X_train[y_train == 0]`).
<br><span class="en">**Isolation Forest**: builds many RANDOM decision trees, each splitting data
by randomly picking a feature + cut threshold — an ANOMALOUS point gets "isolated" (split off
alone) AFTER FEWER cuts than a normal point (since it's clearly different), so the average depth
needed to isolate a point becomes its anomaly score. **LOF**: compares a point's local neighbour
DENSITY against the density of its neighbours' own neighbours — a point sitting in a region much
SPARSER than its surroundings is anomalous. **Autoencoder**: a neural network learns to COMPRESS
then RECONSTRUCT its own input (encoder→bottleneck→decoder) — trained only on normal data, so it
reconstructs normal data WELL but reconstructs abnormal data POORLY (never seen that shape before)
— a large reconstruction error (MSE) = a suspicious point. All 3 are trained ONLY on the majority
class (`X_train[y_train == 0]`).</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Huấn luyện CHỈ trên lớp đa số (income <=50K)
X_majority = X_train[y_train == 0]

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_majority)
iso_scores = -iso.decision_function(X_val)          # điểm càng cao càng bất thường

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True).fit(X_majority)
lof_scores = -lof.score_samples(X_val)

# Autoencoder — sai số tái tạo lớn = bất thường
autoencoder.fit(X_majority, X_majority, epochs=50, batch_size=256)
mse = np.mean(np.square(X_val - autoencoder.predict(X_val)), axis=1)
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk4-notebook]]/"Unsupervised Anomaly Detection": đánh giá bằng 1 "AUPRC-kiểu" tự viết
(`calculate_auprc`) sau khi chuyển điểm bất thường thành nhãn nhị phân qua ngưỡng percentile 5%
— kết quả Isolation Forest (0.6161) nhỉnh hơn LOF (0.6082) và Autoencoder (0.5753). Lưu ý: vì
`precision_recall_curve` được truyền NHÃN NHỊ PHÂN (đã qua ngưỡng) thay vì ĐIỂM SỐ LIÊN TỤC gốc,
đường cong Precision-Recall thực chất chỉ còn vài điểm rời rạc — con số "AUPRC" này kém tin cậy
hơn AUPRC tính đúng cách trên [[auprc]] (dùng điểm số liên tục qua `predict_proba`).
<br><span class="en">[[wk4-notebook]]/"Unsupervised Anomaly Detection": evaluated with a
hand-written "AUPRC-like" function (`calculate_auprc`) after converting anomaly scores to binary
labels via a 5th-percentile threshold — Isolation Forest (0.6161) edges out LOF (0.6082) and the
Autoencoder (0.5753). Note: because `precision_recall_curve` is fed BINARY (already-thresholded)
labels instead of the original CONTINUOUS scores, the Precision-Recall curve effectively collapses
to just a few discrete points — this "AUPRC" figure is less reliable than [[auprc]] computed
properly (using continuous scores via `predict_proba`).</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác phân loại giám sát (XGBoost, [[imbalanced-data-resampling]]) — anomaly detection không cần
2 lớp cân bằng, thậm chí không cần biết trước lớp "bất thường" tồn tại dưới dạng gì; đây là
hướng THAY THẾ khi resampling không khả thi (lớp thiểu số quá hiếm để sinh mẫu tổng hợp đáng
tin). Liên quan [[kmeans-dbscan-clustering]] — DBSCAN's điểm "nhiễu" (label -1) về khái niệm gần
với phát hiện dị thường, dù DBSCAN thiết kế để PHÂN CỤM chứ không tối ưu riêng cho phát hiện bất
thường.
<br><span class="en">Different from supervised classification (XGBoost,
[[imbalanced-data-resampling]]) — anomaly detection doesn't need 2 balanced classes, doesn't even
need to know in advance what form the "abnormal" class takes; it's an ALTERNATIVE direction when
resampling isn't feasible (the minority class too rare to generate trustworthy synthetic
samples). Related to [[kmeans-dbscan-clustering]] — DBSCAN's "noise" points (label -1) are
conceptually close to anomaly detection, though DBSCAN is designed for CLUSTERING, not optimised
specifically for anomaly detection.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[kmeans-dbscan-clustering]] · [[auprc]] · [[imbalanced-data-resampling]]
