---
type: concept
title: "Hierarchical Clustering for Categorical Grouping"
tags: [feature-engineering, clustering, wk2]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

Dùng phân cụm phân cấp (`scipy.cluster.hierarchy.linkage`, thường với `method='ward'`) để nhóm
các mức của MỘT biến phân loại nhiều mức lại với nhau, dựa trên một chỉ số liên tục đo hành vi
của từng mức đối với target (thường là tỷ lệ lớp dương).
<br><span class="en">Uses hierarchical clustering (`scipy.cluster.hierarchy.linkage`, typically
with `method='ward'`) to group the levels of ONE high-cardinality categorical variable together,
based on a continuous metric measuring each level's behaviour relative to the target (usually
the positive-class rate).</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Biến phân loại nhiều mức (`occupation` có 15 mức) nếu one-hot toàn bộ sẽ tạo 15 cột thưa, nhiều
cột trong đó gần như trùng hành vi với nhau đối với target (vd 2 nghề có tỷ lệ thu nhập cao gần
bằng nhau không cần tách 2 cột riêng). Phân cụm dựa trên hành vi thực tế giúp GIẢM số chiều mà
vẫn giữ được tín hiệu quan trọng nhất, đồng thời cho ra các nhóm CÓ Ý NGHĨA (có thể đặt tên, dễ
diễn giải) thay vì các thành phần trừu tượng như PCA.
<br><span class="en">A high-cardinality categorical variable (`occupation` has 15 levels), if
fully one-hot encoded, produces 15 sparse columns, several of which behave nearly identically
relative to the target (e.g. two occupations with near-equal high-income rates don't need two
separate columns). Clustering based on actual behaviour reduces dimensionality while keeping the
most important signal, and yields MEANINGFUL groups (nameable, interpretable) unlike abstract
components such as PCA.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

1. Với mỗi mức của biến phân loại, tính 1 chỉ số tóm tắt hành vi (vd tỷ lệ `income == '>50K'`).
   <br><span class="en">1. For each level of the categorical variable, compute one summary
   metric of its behaviour (e.g. rate of `income == '>50K'`).</span>
2. `linkage(chỉ_số, method='ward')` xây cây phân cụm dựa trên khoảng cách giữa các chỉ số đó.
   <br><span class="en">2. `linkage(metric, method='ward')` builds a cluster tree based on
   distances between those metric values.</span>
3. Vẽ dendrogram, NHÌN BẰNG MẮT để chọn số nhóm hợp lý (không có ngưỡng tự động — quyết định
   mang tính chủ quan có cơ sở).
   <br><span class="en">3. Plot a dendrogram, VISUALLY choose a sensible number of groups (no
   automatic threshold — a subjective-but-informed decision).</span>
4. Gán nhãn nhóm thủ công dựa trên các mức nằm gần nhau trên dendrogram (vd nhóm
   `occupation_cleaner` gồm `Priv-house-serv`, `Handlers-cleaners`, `Other-service`).
   <br><span class="en">4. Manually label groups based on which levels sit close together on
   the dendrogram (e.g. the `occupation_cleaner` group bundles `Priv-house-serv`,
   `Handlers-cleaners`, `Other-service`).</span>

## Ví dụ - <span class="en">Example</span>

```python
from scipy.cluster.hierarchy import linkage, dendrogram

occ_stats = (occupation_data.groupby('occupation')['income']
             .apply(lambda x: (x == '>50K').mean())
             .reset_index(name='high_income_ratio'))

linked = linkage(occ_stats[['high_income_ratio']], method='ward')
dendrogram(linked, orientation='right', labels=occ_stats['occupation'].values)
# -> nhìn dendrogram, gộp thủ công thành 7 nhóm: cleaner, repair, admin, handwork,
#    executive, support, army
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk2-notebook]] — áp dụng cho `occupation` (7 nhóm) và cho tổ hợp
`relationship`+`marital-status`. Kết quả các nhóm này trở thành cột đầu vào cố định cho
`CustomFeatureCreator` xuyên suốt WK2–WK4, xem [[sklearn-column-transformer-pipeline]].
<br><span class="en">[[wk2-notebook]] — applied to `occupation` (7 groups) and to the
`relationship`+`marital-status` combination. The resulting groups become fixed input columns
for `CustomFeatureCreator` throughout WK2–WK4, see [[sklearn-column-transformer-pipeline]].</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác **network graph** ([[network-graph-categorical-eda]]) vốn chỉ trực quan hóa tần suất đồng
xuất hiện giữa 2 biến — hierarchical clustering ở đây GOM NHÓM các mức của 1 biến dựa trên hành
vi với TARGET, phục vụ trực tiếp cho việc tạo đặc trưng, không chỉ để nhìn. Khác PCA
([[pca]], WK4) — PCA tạo thành phần trừu tượng (tổ hợp tuyến tính của MỌI biến số), còn kỹ thuật
này chỉ nhóm các MỨC của MỘT biến phân loại, giữ nguyên khả năng diễn giải.
<br><span class="en">Different from the **network graph** ([[network-graph-categorical-eda]])
which only visualises co-occurrence frequency between two variables — hierarchical clustering
here GROUPS the levels of one variable based on behaviour toward the TARGET, directly feeding
feature creation, not just for viewing. Different from PCA ([[pca]], WK4) — PCA creates abstract
components (linear combinations of ALL numeric variables), while this technique only groups the
LEVELS of a SINGLE categorical variable, preserving interpretability.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk2-notebook]] · [[wk2-data-preparation-feature-engineering]] ·
[[sklearn-column-transformer-pipeline]] · [[network-graph-categorical-eda]]
