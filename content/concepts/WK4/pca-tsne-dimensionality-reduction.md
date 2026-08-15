---
type: concept
title: "Giảm Chiều Dữ Liệu: PCA & t-SNE"
tags: [pca, tsne, dimensionality-reduction, unsupervised-learning, wk4]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**PCA** (Principal Component Analysis) và **t-SNE** (t-distributed Stochastic Neighbor Embedding)
đều là kỹ thuật **giảm chiều** (dimensionality reduction) — biến đổi dữ liệu nhiều cột thành ít
cột hơn — nhưng PCA là phép biến đổi TUYẾN TÍNH giữ cấu trúc toàn cục, còn t-SNE là phép biến đổi
PHI TUYẾN giữ cấu trúc cục bộ, chủ yếu dùng để trực quan hóa.
<br><span class="en">**PCA** (Principal Component Analysis) and **t-SNE** (t-distributed
Stochastic Neighbor Embedding) are both **dimensionality reduction** techniques — turning
many-column data into fewer columns — but PCA is a LINEAR transform preserving global structure,
while t-SNE is a NON-LINEAR transform preserving local structure, mainly used for
visualisation.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Dữ liệu sau feature engineering có 19 cột — nhiều cột trong đó có thể MANG THÔNG TIN TRÙNG LẶP
(vd `education-num` và các cờ `occupation_*` phần nào tương quan với nhau). Giảm chiều giúp: (1)
huấn luyện nhanh hơn, ít overfitting hơn với ít chiều hơn; (2) TRỰC QUAN HÓA được dữ liệu nhiều
chiều bằng mắt thường (chỉ có thể vẽ 2D/3D); (3) phát hiện cấu trúc ẩn (nhóm dữ liệu tự nhiên)
trước khi thử phân cụm.
<br><span class="en">Data after feature engineering has 19 columns — many of which may carry
REDUNDANT information (e.g. `education-num` and the `occupation_*` flags partly correlate).
Dimensionality reduction helps: (1) faster training, less overfitting with fewer dimensions; (2)
VISUALISING high-dimensional data by eye (only 2D/3D can be plotted); (3) discovering hidden
structure (natural data groupings) before attempting clustering.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

**PCA**: tìm trục (thành phần chính) theo hướng dữ liệu biến thiên NHIỀU NHẤT, các trục tiếp
theo trực giao (vuông góc) với trục trước và biến thiên giảm dần — giữ K thành phần đầu giữ được
K phần phương sai lớn nhất. Số thành phần chọn theo NGƯỠNG PHƯƠNG SAI GIẢI THÍCH ĐƯỢC (vd 95%) —
notebook cần 10/19 thành phần để đạt 95%. **t-SNE**: với mỗi cặp điểm, tính xác suất "gần nhau"
trong không gian gốc (nhiều chiều) và không gian đích (2D), rồi tối ưu vị trí các điểm trong
không gian đích sao cho 2 phân phối xác suất đó GIỐNG NHAU NHẤT có thể — điểm gần nhau trong dữ
liệu gốc có xu hướng vẫn gần nhau sau khi giảm chiều, nhưng KHOẢNG CÁCH tuyệt đối trên biểu đồ
không mang ý nghĩa định lượng.
<br><span class="en">**PCA**: finds axes (principal components) along the direction of GREATEST
data variation, subsequent axes orthogonal to previous ones with decreasing variation — keeping
the top K components retains the K largest chunks of variance. Component count is chosen by an
EXPLAINED-VARIANCE THRESHOLD (e.g. 95%) — the notebook needs 10/19 components to reach 95%.
**t-SNE**: for every pair of points, computes a "closeness" probability in the original
(high-dimensional) space and the target (2D) space, then optimises point positions in the target
space so those two probability distributions are AS SIMILAR AS POSSIBLE — points close together
in the original data tend to stay close after reduction, but absolute DISTANCES on the plot carry
no reliable quantitative meaning.</span>

## Ví dụ - <span class="en">Example</span>

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA — chọn số thành phần theo % phương sai giải thích được
pca = PCA().fit(Train_X_transformed)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_n = np.argmax(cumulative_variance >= 0.95) + 1   # = 10 trong notebook gốc

pca_optimal = PCA(n_components=optimal_n)
Train_X_pca = pca_optimal.fit_transform(Train_X_transformed)   # (35065, 10)

# t-SNE — chỉ để trực quan hóa, KHÔNG dùng làm input cho model
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Train_X_tsne = tsne.fit_transform(Train_X_transformed)
plt.scatter(Train_X_tsne[:, 0], Train_X_tsne[:, 1], c=Train_y_transformed, cmap='viridis')
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Xuất hiện ở [[wk4-notebook]]/"Modelling Part IV" — PCA dùng cả để trực quan hóa (heatmap loadings,
scatter PC2 vs PC3) LẪN làm 1 trong 6 chiến lược của lưới 54 tổ hợp (`n_components=9`, khác với
10 thành phần tìm được ở bước khám phá 95% phương sai — 1 điểm không khớp hoàn toàn đáng chú ý).
t-SNE chỉ dùng để trực quan hóa (thử cả trên dữ liệu gốc và sau PCA), không tham gia lưới so
sánh model.
<br><span class="en">Appears in [[wk4-notebook]]/"Modelling Part IV" — PCA is used both for
visualisation (loadings heatmap, PC2 vs PC3 scatter) AND as one of the 54-combination grid's 6
strategies (`n_components=9`, differing from the 10 components found in the 95%-variance
exploration step — a notable, not-fully-matching detail). t-SNE is used only for visualisation
(tried on both raw data and after PCA), not part of the model comparison grid.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

PCA vs t-SNE: PCA nhanh, có công thức toán tường minh, TỔNG QUÁT HÓA được cho dữ liệu mới
(`.transform()`), phù hợp làm bước tiền xử lý cho model; t-SNE chậm hơn nhiều, không có công thức
đóng để áp dụng cho điểm mới (phải chạy lại toàn bộ), chỉ phù hợp trực quan hóa khám phá. Cả 2
khác [[kmeans-dbscan-clustering]] — giảm chiều biến đổi TRỤC TỌA ĐỘ của dữ liệu, còn phân cụm gán
NHÃN NHÓM cho từng điểm; thường dùng NỐI TIẾP nhau (giảm chiều trước, phân cụm/trực quan sau).
<br><span class="en">PCA vs t-SNE: PCA is fast, has an explicit mathematical formula,
GENERALISES to new data (`.transform()`), suits use as a model preprocessing step; t-SNE is much
slower, has no closed-form mapping for new points (must rerun entirely), suits only exploratory
visualisation. Both differ from [[kmeans-dbscan-clustering]] — dimensionality reduction
transforms the data's COORDINATE AXES, while clustering assigns a GROUP LABEL to each point; they
are often used IN SEQUENCE (reduce dimensions first, then cluster/visualise).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk4-notebook]] · [[wk4-model-selection-explainability-fairness-deployment]] ·
[[kmeans-dbscan-clustering]] · [[anomaly-detection]] · [[xgboost]]
