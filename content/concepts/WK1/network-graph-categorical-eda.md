---
type: concept
title: "Network Graph for Categorical EDA"
tags: [eda, networkx, visualization, wk1]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

Dùng đồ thị có hướng (`networkx.DiGraph`) để trực quan hóa mối quan hệ giữa 2 biến phân loại —
mỗi giá trị riêng biệt của biến là 1 node, tần suất xuất hiện đồng thời của 1 cặp giá trị là
trọng số cạnh nối chúng.
<br><span class="en">Uses a directed graph (`networkx.DiGraph`) to visualise the relationship
between two categorical variables — each distinct value of a variable is a node, and the
co-occurrence frequency of a value pair becomes the weight of the edge connecting them.</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Với biến phân loại có nhiều mức (vd `education` 16 mức, `occupation` 15 mức), một bảng chéo
(`crosstab`) 16×15 rất khó đọc trực quan. Đồ thị mạng nén thông tin đó thành hình ảnh — độ dày
cạnh cho biết ngay cặp giá trị nào phổ biến nhất mà không cần dò từng ô trong bảng.
<br><span class="en">For categorical variables with many levels (e.g. `education` has 16
levels, `occupation` has 15), a 16×15 crosstab is hard to read at a glance. A network graph
compresses that into a picture — edge thickness immediately shows which value pairs are most
common, without scanning individual table cells.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

1. `groupby([col_a, col_b]).size()` → đếm tần suất từng cặp.
   <br><span class="en">1. `groupby([col_a, col_b]).size()` → count each pair's frequency.</span>
2. Lọc lấy top-N cặp phổ biến nhất (tránh đồ thị rối vì quá nhiều cạnh).
   <br><span class="en">2. Keep only the top-N most frequent pairs (avoids a cluttered graph
   from too many edges).</span>
3. `G.add_edge(a, b, weight=count)` cho từng cặp, `nx.spring_layout()` để tự động sắp xếp node,
   `nx.draw()` + `nx.draw_networkx_edge_labels()` để vẽ.
   <br><span class="en">3. `G.add_edge(a, b, weight=count)` for each pair, `nx.spring_layout()`
   to auto-arrange nodes, `nx.draw()` + `nx.draw_networkx_edge_labels()` to render.</span>

## Ví dụ - <span class="en">Example</span>

```python
import networkx as nx
import matplotlib.pyplot as plt

top10 = (input_features.loc[:, ['education', 'occupation']].dropna()
         .groupby(['education', 'occupation']).size()
         .reset_index(name='weight')
         .sort_values('weight', ascending=False).head(10))

G = nx.DiGraph()
for _, row in top10.iterrows():
    G.add_edge(row['education'], row['occupation'], weight=row['weight'])

pos = nx.spring_layout(G, k=0.5, iterations=20)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=5000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

[[wk1-notebook]] mục 3.3.1 (cặp `education`↔`occupation` phổ biến nhất). [[wk1-practice]] mở
rộng kỹ thuật này thành **5 cụm mạng riêng biệt** (nhân khẩu học, hành vi tài chính, kênh
marketing, cường độ chiến dịch, mạng lõi với target) cho bộ dữ liệu Bank Marketing.
<br><span class="en">[[wk1-notebook]] section 3.3.1 (most common `education`↔`occupation`
pairs). [[wk1-practice]] extends the technique into **5 separate network clusters**
(demographics, financial behaviour, marketing channel, campaign intensity, core network with the
target) for the Bank Marketing dataset.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Khác với **hierarchical clustering** (xem [[wk2-data-preparation-feature-engineering]]) vốn dùng
để GOM NHÓM các mức của 1 biến phân loại theo một chỉ số liên tục (vd tỷ lệ thu nhập cao) —
network graph ở đây chỉ TRỰC QUAN HÓA tần suất đồng xuất hiện, không phân cụm hay đưa ra quyết
định gộp nhóm nào.
<br><span class="en">Different from **hierarchical clustering** (see
[[wk2-data-preparation-feature-engineering]]) which GROUPS levels of a categorical variable by a
continuous metric (e.g. high-income ratio) — the network graph here only VISUALISES
co-occurrence frequency, it does not cluster or decide which groups to merge.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk1-notebook]] · [[wk1-data-understanding]] · [[wk1-practice]] ·
[[wk2-data-preparation-feature-engineering]]
