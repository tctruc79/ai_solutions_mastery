---
type: concept
title: "Phân Cụm Không Giám Sát: K-Means & DBSCAN"
tags: [kmeans, dbscan, clustering, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**K-Means** và **DBSCAN** đều là thuật toán **phân cụm** (clustering) không giám sát — tự động
nhóm các điểm dữ liệu GIỐNG NHAU lại với nhau mà KHÔNG dùng nhãn `income` — nhưng theo 2 triết lý
khác hẳn nhau: K-Means chia không gian thành K vùng cầu quanh tâm cụm, DBSCAN nhóm theo MẬT ĐỘ
điểm.
<br><span class="en">**K-Means** and **DBSCAN** are both unsupervised **clustering**
algorithms — automatically grouping SIMILAR data points together WITHOUT using the `income` label
— but follow 2 entirely different philosophies: K-Means partitions the space into K spherical
regions around centroids, DBSCAN groups by point DENSITY.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Sau khi đã có model phân loại giám sát tốt (XGBoost), câu hỏi khác đáng đặt ra: dữ liệu có tự
NHIÊN chia thành các NHÓM NGƯỜI khác nhau không, không cần biết trước nhãn thu nhập? Phân cụm
giúp khám phá cấu trúc ẩn này — vd 1 cụm có thể là "trẻ, ít giờ làm, thu nhập thấp" trong khi cụm
khác là "lớn tuổi, học vấn cao, thu nhập cao" — hữu ích cho phân khúc (segmentation) hoặc hiểu dữ
liệu sâu hơn ngoài mục tiêu dự đoán thuần túy.
<br><span class="en">After already having a good supervised classifier (XGBoost), another
worthwhile question: does the data naturally split into different GROUPS OF PEOPLE, without
knowing the income label in advance? Clustering helps uncover this hidden structure — e.g. one
cluster might be "young, few hours worked, low income" while another is "older, highly educated,
high income" — useful for segmentation or deeper data understanding beyond pure prediction.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

**K-Means**: chọn trước số cụm K, khởi tạo K tâm cụm ngẫu nhiên, LẶP 2 bước tới hội tụ: (1) gán
mỗi điểm vào tâm cụm GẦN NHẤT, (2) cập nhật tâm cụm = trung bình các điểm vừa gán. Chọn K bằng
**elbow method** — vẽ tổng bình phương khoảng cách trong cụm (inertia) theo từng K, chọn điểm
"gãy khuỷu tay" nơi thêm cụm không còn giảm inertia đáng kể. **DBSCAN**: với mỗi điểm, đếm số
láng giềng trong bán kính `eps` — nếu đủ `min_samples` láng giềng, điểm đó là **core point** và
bắt đầu/mở rộng 1 cụm; điểm không đủ láng giềng nhưng nằm gần 1 core point là **border point**;
điểm còn lại là **noise** (nhãn `-1`) — DBSCAN tự xác định số cụm, không cần K.
<br><span class="en">**K-Means**: choose the cluster count K upfront, initialise K random
centroids, ITERATE 2 steps to convergence: (1) assign each point to its NEAREST centroid, (2)
update each centroid = mean of its assigned points. K is chosen via the **elbow method** —
plotting within-cluster sum of squares (inertia) against K, picking the "elbow" point where
adding more clusters no longer meaningfully reduces inertia. **DBSCAN**: for each point, count
neighbours within radius `eps` — if it has enough (`min_samples`) neighbours, it's a **core
point** and starts/extends a cluster; a point with too few neighbours but near a core point is a
**border point**; everything else is **noise** (label `-1`) — DBSCAN determines the cluster count
itself, no K needed.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.cluster import KMeans, DBSCAN

# K-Means — elbow method để chọn K
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
            for k in range(1, 25)]
plt.plot(range(1, 25), inertias)   # tìm điểm "gãy khuỷu tay"

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Train_X_transformed)
cluster_income_rate = pd.Series(Train_y_transformed).groupby(kmeans.labels_).mean()

# DBSCAN — không cần K, tự phát hiện nhiễu
dbscan = DBSCAN(eps=0.5, min_samples=40)
labels = dbscan.fit_predict(Train_X_transformed)
n_noise = (labels == -1).sum()   # số điểm bị coi là nhiễu, không thuộc cụm nào
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk4-notebook]]/"Modelling Part IV": K-Means dùng elbow method rồi CHỌN `optimal_k = 10` — dù
comment trong code lại ghi "let's say it appears to be 3" (không khớp với code thực thi — xem
[[wk4-notebook]] mục quirk). DBSCAN dùng `eps=0.5, min_samples=40`, loại bỏ điểm nhiễu trước khi
tính tỷ lệ `>50K` trung bình theo từng cụm còn lại. Cả 2 chỉ dùng để khám phá, không đưa kết quả
cụm vào lưới so sánh 54 tổ hợp của model phân loại.
<br><span class="en">[[wk4-notebook]]/"Modelling Part IV": K-Means uses the elbow method then
SELECTS `optimal_k = 10` — despite a code comment reading "let's say it appears to be 3" (not
matching the code that actually runs — see [[wk4-notebook]]'s quirks section). DBSCAN uses
`eps=0.5, min_samples=40`, dropping noise points before computing the average `>50K` rate per
remaining cluster. Both are used only for exploration, not fed into the 54-combination
classification-model comparison grid.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

K-Means vs DBSCAN: K-Means nhanh, dễ hiểu, nhưng cần biết K trước và giả định cụm hình cầu đều
(không phù hợp cụm hình dạng bất thường); DBSCAN linh hoạt hơn về hình dạng cụm và tự phát hiện
nhiễu/dị thường, nhưng nhạy cảm với việc chọn `eps`/`min_samples` và không xử lý tốt cụm có mật
độ CHÊNH LỆCH lớn. Khác [[hierarchical-clustering-categorical-grouping]] (WK2) — WK2 phân cụm
CÁC MỨC của 1 biến phân loại (occupation) dựa trên tỷ lệ target, còn K-Means/DBSCAN ở đây phân
cụm TỪNG DÒNG DỮ LIỆU (từng người) dựa trên toàn bộ đặc trưng số. Liên quan tới
[[anomaly-detection]] — DBSCAN's khái niệm "nhiễu" gần với ý tưởng phát hiện dị thường, dù mục
đích thiết kế khác nhau (phân cụm vs phát hiện điểm bất thường).
<br><span class="en">K-Means vs DBSCAN: K-Means is fast, easy to understand, but needs K known
upfront and assumes roughly spherical clusters (not suited to irregularly shaped clusters);
DBSCAN is more flexible about cluster shape and self-detects noise/outliers, but is sensitive to
the `eps`/`min_samples` choice and handles clusters of very DIFFERENT density poorly. Different
from [[hierarchical-clustering-categorical-grouping]] (WK2) — WK2 clusters the LEVELS of one
categorical variable (occupation) based on the target rate, while K-Means/DBSCAN here cluster
INDIVIDUAL DATA ROWS (people) based on the full numeric feature set. Related to
[[anomaly-detection]] — DBSCAN's "noise" concept is close to the anomaly-detection idea, though
designed for a different purpose (clustering vs flagging outlier points).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[pca-tsne-dimensionality-reduction]] · [[anomaly-detection]] ·
[[hierarchical-clustering-categorical-grouping]]
