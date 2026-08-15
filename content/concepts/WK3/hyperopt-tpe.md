---
type: concept
title: "Hyperopt & TPE (Tree-structured Parzen Estimator)"
tags: [hyperparameter-tuning, hyperopt, tpe, wk3]
created: 2026-08-14
updated: 2026-08-14
status: complete
---

## Định nghĩa - <span class="en">Definition</span>

**Hyperopt** là thư viện Python tối ưu siêu tham số (hyperparameter tuning); thuật toán mặc định
dùng trong môn học là **TPE** (Tree-structured Parzen Estimator) — một phương pháp tìm kiếm kiểu
Bayes chọn bộ siêu tham số tiếp theo THÔNG MINH dựa trên kết quả các lần thử trước, thay vì duyệt
lưới đều (grid search) hay chọn hoàn toàn ngẫu nhiên (random search).
<br><span class="en">**Hyperopt** is a Python hyperparameter-tuning library; the algorithm used
by default in the course is **TPE** (Tree-structured Parzen Estimator) — a Bayesian-style search
method that INTELLIGENTLY picks the next hyperparameter set based on previous trial results,
instead of sweeping an even grid (grid search) or picking purely at random (random search).</span>

## Trực giác / tại sao cần - <span class="en">Intuition / Why It Matters</span>

XGBoost có nhiều siêu tham số LIÊN TỤC (`learning_rate`, `subsample`, `colsample_bytree`...) —
grid search cần rời rạc hóa từng tham số thành vài mức rồi duyệt TÍCH của tất cả tổ hợp, chi phí
tăng theo cấp số nhân với số tham số. Random search rẻ hơn nhưng không học được gì từ các lần thử
trước — có thể lãng phí thời gian thử lại những vùng đã biết là kém. TPE cân bằng cả 2: rẻ như
random search (không cần duyệt lưới đầy đủ) nhưng vẫn HỌC từ lịch sử như 1 phương pháp có định
hướng.
<br><span class="en">XGBoost has many CONTINUOUS hyperparameters (`learning_rate`, `subsample`,
`colsample_bytree`...) — grid search needs to discretise each into a few levels then sweep the
PRODUCT of all combinations, with cost growing exponentially in the number of parameters. Random
search is cheaper but learns nothing from previous trials — it can waste time re-trying regions
already known to be poor. TPE balances both: as cheap as random search (no full grid sweep
needed) yet still LEARNS from history like a directed method.</span>

## Cơ chế hoạt động - <span class="en">Mechanism</span>

TPE chia các lần thử đã có thành 2 nhóm dựa trên kết quả (tốt / kém hơn 1 ngưỡng), rồi mô hình
hóa PHÂN PHỐI của siêu tham số trong mỗi nhóm — vùng tham số THƯỜNG xuất hiện ở nhóm "tốt" và
HIẾM ở nhóm "kém" được ưu tiên thử tiếp. Trong môn học: `hp.quniform('n_estimators', 50, 300,
10)` định nghĩa không gian tìm kiếm rời rạc bước 10; `hp.uniform('learning_rate', 0.01, 0.3)`
định nghĩa không gian liên tục; hàm `objective(params)` trả về `{'loss': -score, 'status':
STATUS_OK}` (Hyperopt LUÔN minimize, nên phải ÂM HÓA AUPRC vì AUPRC càng cao càng tốt);
`fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50, trials=Trials())` chạy 50
lần thử, `Trials()` lưu lại lịch sử để TPE học dần qua từng vòng.
<br><span class="en">TPE splits existing trials into 2 groups by outcome (good / worse than a
threshold), then models the DISTRIBUTION of hyperparameters within each group — parameter regions
that appear OFTEN in the "good" group and RARELY in the "poor" group are prioritised for the next
trial. In the course: `hp.quniform('n_estimators', 50, 300, 10)` defines a discrete search space
with step 10; `hp.uniform('learning_rate', 0.01, 0.3)` defines a continuous space; the
`objective(params)` function returns `{'loss': -score, 'status': STATUS_OK}` (Hyperopt ALWAYS
minimises, so AUPRC — where higher is better — must be NEGATED); `fmin(fn=objective,
space=param_space, algo=tpe.suggest, max_evals=50, trials=Trials())` runs 50 trials, with
`Trials()` storing history so TPE learns progressively across rounds.</span>

## Ví dụ - <span class="en">Example</span>

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

def objective(params):
    model = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
    )
    score = cross_val_score(model, X_train, y_train, cv=skf,
                             scoring='average_precision').mean()
    return {'loss': -score, 'status': STATUS_OK}   # âm hoá vì Hyperopt luôn minimize

param_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
}

best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest,
                    max_evals=50, trials=Trials())
```

## Xuất hiện ở đâu trong môn - <span class="en">Where It Appears</span>

Dùng trong [[wk3-notebook]]/"XGBoost for Imbalanced Data" để tinh chỉnh XGBoost trên tập đã
SMOTE-resample (xem [[imbalanced-data-resampling]]) — nâng AUPRC từ 0.959 (tham số mặc định) lên
0.9635 sau 50 lần thử (~37 phút chạy thực tế), model cuối cùng dùng để so sánh trong
[[wk4-notebook]].
<br><span class="en">Used in [[wk3-notebook]]/"XGBoost for Imbalanced Data" to fine-tune XGBoost
on the SMOTE-resampled set (see [[imbalanced-data-resampling]]) — raising AUPRC from 0.959
(default parameters) to 0.9635 after 50 trials (~37 minutes of actual runtime), the resulting
model used for comparison in [[wk4-notebook]].</span>

## So sánh với khái niệm liên quan - <span class="en">Related Concepts</span>

**Grid Search**: chắc chắn tìm được điểm tốt nhất TRONG lưới đã định nghĩa, nhưng chi phí tăng
theo cấp số nhân — không khả thi với nhiều tham số liên tục. **Random Search**: rẻ, dễ triển
khai, nhưng không tận dụng thông tin từ các lần thử trước. **TPE (Hyperopt)**: cân bằng giữa 2
thái cực trên — học từ lịch sử để tập trung tìm kiếm vào vùng có triển vọng, phù hợp khi không
gian tham số lớn/liên tục và mỗi lần thử tốn kém (như huấn luyện XGBoost qua 10-fold CV).
<br><span class="en">**Grid Search**: guaranteed to find the best point WITHIN the defined grid,
but cost grows exponentially — infeasible with many continuous parameters. **Random Search**:
cheap, easy to implement, but doesn't leverage information from previous trials. **TPE
(Hyperopt)**: balances the two extremes above — learns from history to focus the search on
promising regions, well suited when the parameter space is large/continuous and each trial is
expensive (like training XGBoost through 10-fold CV).</span>

## Liên kết wiki - <span class="en">Wiki Links</span>

[[wk3-notebook]] · [[wk3-modelling-imbalance-deep-learning]] · [[xgboost]] ·
[[imbalanced-data-resampling]] · [[auprc]]
