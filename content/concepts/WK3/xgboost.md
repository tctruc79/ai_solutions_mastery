---
type: concept
title: "XGBoost"
tags: [algorithm, gradient-boosting, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**XGBoost** (Extreme Gradient Boosting) là thuật toán ensemble cây xây TUẦN TỰ — mỗi cây mới
được huấn luyện để sửa lỗi (gradient của hàm mất mát) mà các cây trước đó còn để lại, khác hẳn
Random Forest xây cây độc lập song song ([[random-forest]]).
<br><span class="en">**XGBoost** (Extreme Gradient Boosting) is a SEQUENTIAL tree-ensemble
algorithm — each new tree is trained to correct the errors (loss-function gradient) left by the
previous trees, unlike Random Forest which builds trees independently in parallel
([[random-forest]]).</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

Vì mỗi cây "học từ lỗi" của các cây trước, boosting thường đạt độ chính xác cao hơn bagging
(Random Forest) trên cùng dữ liệu, đặc biệt trên bài toán khó/mất cân bằng — đúng như kết quả
môn học: XGBoost (0.7976) và Gradient Boosting (0.7962) dẫn đầu bảng so sánh 9 model, vượt xa
Random Forest (0.7124) trên dữ liệu gốc.
<br><span class="en">Because each tree "learns from the errors" of previous trees, boosting
typically achieves higher accuracy than bagging (Random Forest) on the same data, especially on
hard/imbalanced problems — exactly matching the course's result: XGBoost (0.7976) and Gradient
Boosting (0.7962) lead the 9-model comparison, well ahead of Random Forest (0.7124) on the
original data.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

2 cách dùng trong môn học: (1) API sklearn-style `XGBClassifier(...)` — quen thuộc, dễ dùng
trong `cross_val_score`/`Pipeline`; (2) API gốc `xgb.DMatrix` + `xgb.train(params, dtrain,
custom_metric=..., evals=...)` — cho phép truyền hàm đánh giá TÙY BIẾN (vd `auprc_eval` tự viết)
theo dõi mỗi vòng boosting, và tham số `scale_pos_weight = count(lớp 0) / count(lớp 1)` để mô
hình tự cân bằng phần nào ngay trong quá trình huấn luyện, không cần resampling trước.
<br><span class="en">2 usages in the course: (1) the sklearn-style API `XGBClassifier(...)` —
familiar, easy to drop into `cross_val_score`/`Pipeline`; (2) the native API `xgb.DMatrix` +
`xgb.train(params, dtrain, custom_metric=..., evals=...)` — allows passing a CUSTOM evaluation
function (e.g. a hand-written `auprc_eval`) tracked every boosting round, and the
`scale_pos_weight = count(class 0) / count(class 1)` parameter so the model self-balances
somewhat during training, without needing resampling first.</span>

## Ví dụ - <span class="en">Example</span>

```python
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

def auprc_eval(preds, dtrain):
    labels = dtrain.get_label()
    precision, recall, _ = precision_recall_curve(labels, preds)
    return 'auprc', float(auc(recall, precision))

scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
params = {'objective': 'binary:logistic', 'scale_pos_weight': scale_pos_weight,
          'eval_metric': 'logloss', 'seed': 42}

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=100,
                   evals=[(dtrain, 'train')], custom_metric=auprc_eval)

feature_importance = model.get_score(importance_type='gain')  # quan trọng hơn 'weight'/'cover'
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Trở thành model TRUNG TÂM từ [[wk3-notebook]] trở đi: đào sâu ở WK3 (kết hợp
[[imbalanced-data-resampling]] + [[hyperopt-tpe]]), tiếp tục là ứng viên hàng đầu trong grid 54
tổ hợp ở [[wk4-notebook]] (thắng cuộc với XGBoost + CTGAN Oversampling, AUPRC ≈ 0.964 trên CV,
nhưng ~0.80 trên test set thật — bài học quan trọng về đánh giá đúng cách), và là model được
chọn để giải thích bằng [[shap]] và kiểm tra công bằng bằng [[fairlearn]].
<br><span class="en">Becomes the CENTRAL model from [[wk3-notebook]] onward: deep-dived in WK3
(combined with [[imbalanced-data-resampling]] + [[hyperopt-tpe]]), remains the top contender in
the 54-combination grid in [[wk4-notebook]] (winning with XGBoost + CTGAN Oversampling, AUPRC ≈
0.964 on CV, but ~0.80 on the real test set — an important lesson on proper evaluation), and is
the model chosen for [[shap]] explainability and [[fairlearn]] fairness checks.</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

Xem [[random-forest]] để so sánh bagging vs boosting (Random Forest đạt 0.7141 AUPRC trên dữ
liệu gốc trong cùng bảng so sánh 9 model — thấp hơn rõ rệt cả XGBoost lẫn Gradient Boosting). So
với **Gradient Boosting** (sklearn gốc, cũng có mặt trong bảng 9 model) — XGBoost thêm
regularization (L1/L2), xử lý giá trị thiếu tự động, và tối ưu tốc độ huấn luyện hơn — 2 thuật
toán cho kết quả gần nhau (0.7976 vs 0.7962 AUPRC) vì cùng chung nguyên lý boosting.
<br><span class="en">See [[random-forest]] for the bagging-vs-boosting comparison (Random Forest
reaches 0.7141 AUPRC on the original data in the same 9-model comparison — clearly behind both
XGBoost and Gradient Boosting). Versus **Gradient Boosting** (plain sklearn version, also in the
9-model table) — XGBoost adds regularisation (L1/L2), automatic missing-value handling, and
faster training — the two give close results (0.7976 vs 0.7962 AUPRC) since they share the same
boosting principle.</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[random-forest]] · [[auprc]] ·
[[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[wk4-notebook]]
