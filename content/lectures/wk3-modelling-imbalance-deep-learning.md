---
type: lecture
title: "WK3 — Modelling II: So Sánh 9 Model, XGBoost, Xử Lý Mất Cân Bằng & Deep Learning"
tags: [modelling, xgboost, imbalanced-data, hyperopt, deep-learning, ctgan, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
source_file: "raw/Lectures/honeyg_ai-practitioner-wk3-Modelling-9 Supervised Learning-Deep Learning - SMOTE - ADASync - CTGANs.ipynb"
---

## Vị trí trong lộ trình môn học - <span class="en">Position in the Course</span>

Tiếp nối trực tiếp [[wk2-data-preparation-feature-engineering]] — pipeline feature engineering
tuần trước được TÁI SỬ DỤNG NGUYÊN VẸN, và bài học cuối tuần 2 ("accuracy 83-84% là ảo giác khi
mất cân bằng lớp") trở thành ĐIỂM XUẤT PHÁT của cả tuần này: mọi thứ từ đây được thiết kế để trả
lời câu hỏi "vậy đo đúng bằng gì, và xử lý mất cân bằng thế nào?". Tuần 4 sẽ mở rộng thành 1 lưới
9 model × 6 chiến lược xử lý mất cân bằng, dùng lại toàn bộ kỹ thuật học ở đây.
<br><span class="en">Follows directly from [[wk2-data-preparation-feature-engineering]] — last
week's feature-engineering pipeline is REUSED UNCHANGED, and the closing lesson of Week 2 ("83-
84% accuracy is an illusion under class imbalance") becomes this week's STARTING POINT: everything
here is designed to answer "so what's the right metric, and how do we actually fix the
imbalance?". Week 4 expands this into a full grid of 9 models × 6 imbalance-handling strategies,
reusing every technique learned here.</span>

## Mục tiêu học tập - <span class="en">Learning Objectives</span>

Sau bài này, bạn làm được: (1) giải thích vì sao AUPRC phù hợp hơn accuracy/ROC-AUC cho bài toán
mất cân bằng và biết baseline ngẫu nhiên của nó; (2) so sánh khách quan nhiều model bằng
cross-validation trên cùng 1 thước đo; (3) dùng XGBoost ở cả 2 API (sklearn-style và native) với
custom eval metric; (4) phân biệt cơ chế SMOTE/ADASYN/Random Undersampling và biết khi nào mỗi
kỹ thuật hợp lý; (5) dùng Hyperopt (TPE) để tìm siêu tham số thông minh hơn grid search; (6) nhận
diện lỗi kiến trúc mạng neural phổ biến (activation cuối không khớp hàm mất mát) khi thấy
`loss: nan`; (7) hiểu CTGAN sinh dữ liệu tổng hợp khác cơ chế với SMOTE ra sao.
<br><span class="en">After this lesson you can: (1) explain why AUPRC suits an imbalanced
problem better than accuracy/ROC-AUC, and know its random baseline; (2) objectively compare
multiple models via cross-validation on the same metric; (3) use XGBoost through both APIs
(sklearn-style and native) with a custom eval metric; (4) distinguish the mechanisms of
SMOTE/ADASYN/Random Undersampling and know when each is appropriate; (5) use Hyperopt (TPE) for
smarter hyperparameter search than grid search; (6) recognise a common neural-network
architecture bug (final activation mismatched with the loss function) when you see `loss: nan`;
(7) understand how CTGAN's synthetic-data mechanism differs from SMOTE's.</span>

## Lý thuyết nền tảng - <span class="en">Core Theory</span>

### 1. Vì sao đổi hẳn sang AUPRC làm thước đo chính thức

WK2 dừng lại ở nhận xét "accuracy gây hiểu lầm"; WK3 hành động cụ thể: MỌI so sánh model từ đây
dùng `make_scorer(average_precision_score, needs_proba=True)` — xem [[auprc]]. Lý do sâu hơn:
ngay cả ROC-AUC (vốn được xem là robust hơn accuracy) vẫn có thể "trông tốt" một cách ảo trên dữ
liệu mất cân bằng vì nó tính cả True Negative Rate, mà với lớp âm chiếm 76%, TNR gần như luôn
cao. AUPRC tập trung hoàn toàn vào lớp dương nên trung thực hơn.
<br><span class="en">Week 2 stopped at the observation "accuracy is misleading"; Week 3 acts on
it: EVERY model comparison from here uses `make_scorer(average_precision_score,
needs_proba=True)` — see [[auprc]]. The deeper reason: even ROC-AUC (usually considered more
robust than accuracy) can still look deceptively good on imbalanced data because it factors in
the True Negative Rate, and with the negative class at 76%, TNR stays high almost by default.
AUPRC focuses entirely on the positive class, so it's more honest.</span>

### 2. So sánh 9 model — một quy trình khách quan, lặp lại được

Thay vì huấn luyện rời rạc từng model rồi so bằng mắt, notebook đóng gói 9 model (Naive Bayes,
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Extra Trees,
AdaBoost, BaggingClassifier) vào 1 dict, chạy CÙNG 1 vòng lặp `cross_val_score` với CÙNG 1
`StratifiedKFold(n_splits=10)` và CÙNG 1 scorer AUPRC — đảm bảo so sánh công bằng (cùng dữ liệu,
cùng cách chia fold, cùng thước đo). Kết quả: XGBoost (0.7976) và Gradient Boosting (0.7962) dẫn
đầu; Decision Tree (0.5069) yếu nhất, sát nút Extra Trees (0.6079).
<br><span class="en">Instead of training each model separately and eyeballing the comparison,
the notebook packages all 9 models (Naive Bayes, Logistic Regression, Decision Tree, Random
Forest, Gradient Boosting, XGBoost, Extra Trees, AdaBoost, BaggingClassifier) into one dict and
runs the SAME `cross_val_score` loop with the SAME `StratifiedKFold(n_splits=10)` and the SAME
AUPRC scorer — guaranteeing a fair comparison (same data, same fold split, same metric). Result:
XGBoost (0.7976) and Gradient Boosting (0.7962) lead; Decision Tree (0.5069) is weakest, close to
Extra Trees (0.6079).</span>

### 3. XGBoost — 2 cách dùng, và vì sao dùng API gốc khi cần kiểm soát sâu

[[xgboost]] có 2 API: `XGBClassifier` (sklearn-style, tiện cho `cross_val_score`/`Pipeline`, dùng
ở bước so sánh 9 model) và API gốc `xgb.DMatrix` + `xgb.train(params, dtrain, custom_metric=...,
evals=...)` (dùng ở phần đào sâu XGBoost) — API gốc cho phép truyền `custom_metric=auprc_eval`
(hàm tự viết, theo dõi AUPRC MỖI vòng boosting qua `evals_result`) và tham số
`scale_pos_weight = count(lớp 0) / count(lớp 1)` để XGBoost tự điều chỉnh trọng số học trong lúc
huấn luyện — một cách xử lý mất cân bằng KHÔNG CẦN resampling trước. Feature importance dùng
`importance_type='gain'` (đóng góp thực vào việc giảm loss) thay vì `'weight'` (số lần cây dùng
đặc trưng đó) — `gain` phản ánh đúng mức độ QUAN TRỌNG hơn là TẦN SUẤT xuất hiện.
<br><span class="en">[[xgboost]] has 2 APIs: `XGBClassifier` (sklearn-style, convenient for
`cross_val_score`/`Pipeline`, used in the 9-model comparison) and the native `xgb.DMatrix` +
`xgb.train(params, dtrain, custom_metric=..., evals=...)` (used in the XGBoost deep-dive) — the
native API allows passing `custom_metric=auprc_eval` (a hand-written function, tracked EVERY
boosting round via `evals_result`) and `scale_pos_weight = count(class 0) / count(class 1)` so
XGBoost self-adjusts its learning weight during training — a way to handle imbalance WITHOUT
resampling first. Feature importance uses `importance_type='gain'` (actual contribution to loss
reduction) rather than `'weight'` (how many times a tree used that feature) — `gain` more
accurately reflects IMPORTANCE rather than FREQUENCY.</span>

### 4. Ba kỹ thuật resampling — cơ chế khác nhau, không chỉ là "cân bằng số lượng"

[[imbalanced-data-resampling]] gồm 3 kỹ thuật cơ chế hoàn toàn khác nhau dù cùng mục đích cân
bằng lớp: **SMOTE** (sinh mẫu thiểu số mới bằng nội suy tuyến tính giữa 1 điểm và láng giềng gần
nhất cùng lớp), **ADASYN** (như SMOTE nhưng sinh NHIỀU mẫu hơn ở vùng biên khó phân loại), và
**Random Undersampling** (bớt ngẫu nhiên mẫu đa số, không sinh gì mới). Kết quả thực nghiệm:
XGBoost + SMOTE đạt AUPRC ≈ 0.959, vượt xa ADASYN (≈0.949) và cả dữ liệu gốc — SMOTE được chọn
(`best_X_train, best_y_train = Train_X_smote, Train_y_smote`) cho bước tuning tiếp theo.
<br><span class="en">[[imbalanced-data-resampling]] covers 3 techniques with entirely different
mechanisms despite sharing the goal of balancing classes: **SMOTE** (generates new minority
samples via linear interpolation between a point and its nearest same-class neighbour),
**ADASYN** (like SMOTE but generates MORE samples in hard-to-classify border regions), and
**Random Undersampling** (randomly drops majority samples, generates nothing new). Empirical
result: XGBoost + SMOTE reaches AUPRC ≈ 0.959, well ahead of ADASYN (≈0.949) and the original
data — SMOTE is chosen (`best_X_train, best_y_train = Train_X_smote, Train_y_smote`) for the next
tuning step.</span>

### 5. Hyperopt (TPE) — tìm kiếm thông minh thay vì duyệt lưới

[[hyperopt-tpe]] dùng thuật toán Tree-structured Parzen Estimator để chọn bộ siêu tham số tiếp
theo DỰA TRÊN kết quả các lần thử trước, thay vì duyệt đều mọi tổ hợp (grid search — chi phí tăng
theo cấp số nhân khi nhiều tham số liên tục) hay chọn ngẫu nhiên hoàn toàn (random search — không
học được gì từ lịch sử). `fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50)`
đưa AUPRC từ 0.959 (SMOTE, tham số mặc định) lên 0.9635 (SMOTE + tuned) sau 50 lần thử — mất
~37 phút chạy trong notebook gốc, cho thấy chi phí tính toán thực tế của tuning không hề nhỏ.
<br><span class="en">[[hyperopt-tpe]] uses the Tree-structured Parzen Estimator algorithm to pick
the next hyperparameter set BASED ON previous trial results, instead of sweeping every
combination evenly (grid search — cost grows exponentially with more continuous parameters) or
picking purely at random (random search — learns nothing from history). `fmin(fn=objective,
space=param_space, algo=tpe.suggest, max_evals=50)` raises AUPRC from 0.959 (SMOTE, default
params) to 0.9635 (SMOTE + tuned) after 50 trials — taking ~37 minutes in the original notebook,
a reminder that tuning has real computational cost.</span>

### 6. Deep Learning — activation lớp cuối PHẢI khớp hàm mất mát

Mô hình Keras đầu tiên dùng `Dense(2, activation='tanh')` (miền giá trị [-1, 1]) làm lớp cuối
nhưng lại compile với `loss='categorical_crossentropy'` (yêu cầu xác suất trong [0, 1]) — kết
hợp SAI này khiến `loss: nan` xuất hiện ngay từ epoch 1 và không bao giờ hồi phục qua cả 20
epoch. Đây là lỗi kỹ thuật THẬT trong notebook gốc, không phải minh họa cố ý — bài học: với phân
loại nhị phân/đa lớp bằng cross-entropy, lớp cuối phải dùng `sigmoid` (nhị phân, 1 output) hoặc
`softmax` (đa lớp, output = số lớp), KHÔNG BAO GIỜ `tanh`. Notebook không sửa cell này mà chuyển
sang kiến trúc khác (`sigmoid` + `binary_crossentropy`, 1 output) ở phần sau.
<br><span class="en">The first Keras model uses `Dense(2, activation='tanh')` (range [-1, 1]) as
its final layer but compiles with `loss='categorical_crossentropy'` (which requires probabilities
in [0, 1]) — this WRONG pairing makes `loss: nan` appear from epoch 1 and never recover across all
20 epochs. This is a genuine technical bug in the original notebook, not intentional
illustration — the lesson: for binary/multi-class classification with cross-entropy, the final
layer must use `sigmoid` (binary, 1 output) or `softmax` (multi-class, output = number of
classes), NEVER `tanh`. The notebook doesn't fix this cell but switches to a different
architecture (`sigmoid` + `binary_crossentropy`, 1 output) afterward.</span>

### 7. CTGAN — sinh dữ liệu tổng hợp bằng mạng đối kháng, khác cơ chế với SMOTE

[[ctgan]] (Conditional Tabular GAN) không nội suy tuyến tính giữa các điểm có thật như SMOTE, mà
huấn luyện 1 cặp mạng neural đối kháng (generator sinh mẫu giả, discriminator cố phân biệt
thật/giả) để HỌC LẠI phân phối dữ liệu, rồi sinh mẫu mới từ phân phối đã học đó — về lý thuyết có
thể tạo ra mẫu đa dạng hơn nội suy tuyến tính, nhưng tốn thời gian huấn luyện hơn nhiều (dù chỉ
`CTGAN(epochs=5)` — rất ít epoch — vẫn là mạng neural cần huấn luyện, khác SMOTE là phép tính
hình học tức thời). Kết quả 1 fold minh họa: CTGAN Oversampling (AUPRC ≈ 0.728) không rõ ràng
vượt trội Base Model (≈ 0.732) trong lần thử này — output đầy đủ 5-fold không được lưu lại trong
notebook gốc, KHÔNG thể kết luận chắc chắn CTGAN thắng hay thua ở bước này chỉ từ 1 fold.
<br><span class="en">[[ctgan]] (Conditional Tabular GAN) doesn't linearly interpolate between
real points like SMOTE — it trains a pair of adversarial neural networks (a generator producing
fake samples, a discriminator trying to tell real from fake) to RELEARN the data distribution,
then samples new points from that learned distribution — in principle this can produce more
diverse samples than linear interpolation, but costs far more training time (even at just
`CTGAN(epochs=5)` — very few epochs — it's still a neural network requiring training, unlike
SMOTE's instant geometric computation). One illustrative fold: CTGAN Oversampling (AUPRC ≈ 0.728)
doesn't clearly beat the Base Model (≈ 0.732) in this run — the full 5-fold output wasn't
preserved in the original notebook, so CTGAN's win/loss at this step CANNOT be concluded from a
single fold alone.</span>

## Ví dụ minh họa tổng hợp - <span class="en">Worked Example</span>

```python
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb_base_params = {'random_state': 42, 'eval_metric': 'logloss',
                    'scale_pos_weight': scale_pos_weight}

# So sánh 3 kỹ thuật resampling bằng AUPRC
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_adasyn, y_adasyn = ADASYN(random_state=42).fit_resample(X_train, y_train)
X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

for name, (X_r, y_r) in {'SMOTE': (X_smote, y_smote), 'ADASYN': (X_adasyn, y_adasyn),
                          'Undersample': (X_under, y_under)}.items():
    score = cross_val_score(xgb.XGBClassifier(**xgb_base_params), X_r, y_r,
                             cv=skf, scoring='average_precision').mean()
    print(f'{name}: {score:.4f}')   # SMOTE thắng rõ rệt trong notebook gốc (~0.959)

# Hyperopt TPE tuning trên chiến lược thắng (SMOTE)
def objective(params):
    model = xgb.XGBClassifier(**xgb_base_params,
                               n_estimators=int(params['n_estimators']),
                               max_depth=int(params['max_depth']),
                               learning_rate=params['learning_rate'])
    score = cross_val_score(model, X_smote, y_smote, cv=skf,
                             scoring='average_precision').mean()
    return {'loss': -score, 'status': STATUS_OK}   # Hyperopt LUÔN minimize -> âm hoá AUPRC

space = {'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
         'max_depth': hp.quniform('max_depth', 3, 15, 1),
         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)}
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials())
```

## Lỗi thường gặp & lưu ý thực hành - <span class="en">Pitfalls & Practical Notes</span>

- **Resample TRƯỚC rồi mới cross-validate** (như notebook gốc làm: `Train_X_smote =
  smote.fit_resample(...)` một lần trên toàn bộ train, sau đó mới `cross_val_score(cv=skf)` trên
  dữ liệu đã resample) — có RỦI RO rò rỉ dữ liệu tinh vi: 1 mẫu tổng hợp SMOTE trong fold validation
  có thể được nội suy từ 1 mẫu thật đang nằm trong fold train, khiến fold validation không còn
  hoàn toàn "chưa từng thấy". Cách làm đúng hơn: đặt SMOTE BÊN TRONG mỗi fold (vd
  `imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', model)])` rồi mới `cross_val_score`
  trên Pipeline đó) để resampling chỉ dùng thông tin của phần train trong fold đó.
  <br><span class="en">**Resampling BEFORE cross-validating** (as the original notebook does:
  `Train_X_smote = smote.fit_resample(...)` once on the whole train set, then
  `cross_val_score(cv=skf)` on the already-resampled data) — carries a SUBTLE leakage risk: a
  SMOTE synthetic sample in the validation fold may have been interpolated from a real sample
  sitting in the train fold, so the validation fold is no longer fully "unseen". The more correct
  approach: put SMOTE INSIDE each fold (e.g. `imblearn.pipeline.Pipeline([('smote', SMOTE()),
  ('clf', model)])`, then `cross_val_score` on that Pipeline) so resampling only uses information
  from that fold's training portion.</span>
- **`Dense(N, activation='tanh')` + `categorical_crossentropy`** — bất kỳ khi nào thấy `loss: nan`
  ngay từ epoch đầu, kiểm tra NGAY sự khớp giữa activation lớp cuối và hàm mất mát trước khi nghi
  ngờ learning rate hay dữ liệu.
  <br><span class="en">**`Dense(N, activation='tanh')` + `categorical_crossentropy`** —
  whenever you see `loss: nan` from the very first epoch, check the match between the final
  activation and the loss function FIRST, before suspecting the learning rate or the data.</span>
- **So sánh model bằng các thước đo khác nhau giữa các bước** (accuracy ở WK2, AUPRC ở WK3) — chỉ
  hợp lệ khi biết RÕ lý do đổi thước đo; đừng so sánh trực tiếp accuracy 83% (WK2) với AUPRC 0.80
  (WK3) như thể chúng cùng thang đo.
  <br><span class="en">**Comparing models across different metrics between steps** (accuracy in
  WK2, AUPRC in WK3) — only valid when you clearly understand WHY the metric changed; don't
  directly compare 83% accuracy (WK2) with 0.80 AUPRC (WK3) as if they were on the same
  scale.</span>

## Liên kết concepts liên quan - <span class="en">Related Concepts</span>

[[auprc]] · [[xgboost]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[random-forest]] · [[logistic-regression]]

## Câu hỏi tự kiểm tra nhanh - <span class="en">Quick Self-Check</span>

1. Vì sao AUPRC phù hợp hơn cả ROC-AUC (chứ không chỉ accuracy) cho bài toán mất cân bằng của
   tuần này?
   <br><span class="en">Why is AUPRC better suited than even ROC-AUC (not just accuracy) for
   this week's imbalanced problem?</span>
2. `scale_pos_weight` trong XGBoost và resampling (SMOTE/ADASYN) đều "xử lý mất cân bằng" — chúng
   khác nhau ở điểm nào trong quy trình huấn luyện?
   <br><span class="en">`scale_pos_weight` in XGBoost and resampling (SMOTE/ADASYN) both "handle
   imbalance" — where in the training process do they actually differ?</span>
3. Mô hình Deep Learning đầu tiên cho `loss: nan` — nguyên nhân kỹ thuật chính xác là gì?
   <br><span class="en">The first Deep Learning model produces `loss: nan` — what's the exact
   technical cause?</span>

Bộ câu hỏi đầy đủ hơn: [[wk3-quiz]].
<br><span class="en">Full question bank: [[wk3-quiz]].</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk2-data-preparation-feature-engineering]] · [[wk2-notebook]] ·
[[auprc]] · [[xgboost]] · [[imbalanced-data-resampling]] · [[hyperopt-tpe]] · [[ctgan]] ·
[[random-forest]] · [[logistic-regression]] · [[overview]]
