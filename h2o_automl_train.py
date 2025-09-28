# # h2o_automl_train.py

# import h2o
# from h2o.automl import H2OAutoML
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import RobustScaler

# # ================================
# # 0) 데이터 로드
# # ================================
# train = pd.read_csv("data/train.csv")
# test  = pd.read_csv("data/test.csv")
# sub   = pd.read_csv("data/sample_submission.csv")

# LABEL = "target"
# ID_COL = "ID"

# # ================================
# # 1) 전처리 (로그 변환 + 스케일링)
# # ================================
# # 로그 변환할 컬럼 (왜도 높은 컬럼 위주)
# LOG_COLS = ["X_11", "X_19", "X_37", "X_40"]
# for col in LOG_COLS:
#     if col in train.columns:
#         train[col] = np.log1p(train[col])
#         test[col]  = np.log1p(test[col])

# # RobustScaler (이상치에 강건)
# scaler = RobustScaler()
# X_cols = [c for c in train.columns if c not in [ID_COL, LABEL]]
# train[X_cols] = scaler.fit_transform(train[X_cols])
# test[X_cols]  = scaler.transform(test[X_cols])

# # ================================
# # 2) H2O 초기화
# # ================================
# h2o.init(max_mem_size="8G")   # 필요 시 조절 가능

# # Pandas → H2OFrame 변환
# train_h2o = h2o.H2OFrame(train)
# test_h2o  = h2o.H2OFrame(test)

# x = [c for c in train.columns if c not in [ID_COL, LABEL]]
# y = LABEL

# train_h2o[y] = train_h2o[y].asfactor()   # 분류 문제라서 factor 변환

# # ================================
# # 3) Feature Importance 기반 컬럼 선택 (GBM 빠르게 학습)
# # ================================
# from h2o.estimators.gbm import H2OGradientBoostingEstimator

# gbm = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, seed=42)
# gbm.train(x=x, y=y, training_frame=train_h2o)

# # 중요도 확인
# importance = gbm.varimp(use_pandas=True)
# selected_features = importance[importance["relative_importance"] > 0.001]["variable"].tolist()

# print(f"선택된 주요 Feature 개수: {len(selected_features)} / {len(x)}")

# # ================================
# # 4) AutoML 실행 (빠른 버전, F1 중심)
# # ================================
# print("\nStarting AutoML training...")

# aml = H2OAutoML(
#     max_runtime_secs=1800,        # 30분 제한
#     max_models=12,                # 모델 수 제한
#     seed=42,
#     nfolds=5,                     # 5-fold CV
#     balance_classes=True,         # 클래스 불균형 처리
#     exclude_algos=[],             # 모든 알고리즘 포함
#     sort_metric="mean_per_class_error",  # H2O는 F1 직접 지원X → error 최소화로 대체
#     stopping_metric="mean_per_class_error",
#     stopping_tolerance=0.001,
#     stopping_rounds=2,
#     keep_cross_validation_predictions=True,
#     keep_cross_validation_models=True
# )

# aml.train(x=selected_features, y=y, training_frame=train_h2o)

# # ================================
# # 5) 결과 분석 (F1 추가 계산)
# # ================================
# print("\nAnalyzing results...")

# # 리더보드 출력
# lb = aml.leaderboard
# print("Top 10 models:")
# print(lb.head(rows=10))

# # 리더 모델로 validation 예측해서 macro-F1 계산
# from sklearn.metrics import f1_score
# valid_preds = aml.leader.predict(train_h2o).as_data_frame()["predict"]
# valid_true = train[LABEL].values
# macro_f1 = f1_score(valid_true, valid_preds, average="macro")
# print(f"\n[Leader Model Macro-F1 on Train] {macro_f1:.5f}")

# # ================================
# # 6) 제출 파일 생성 (단일 + 앙상블 2종)
# # ================================
# print("\nCreating submission files...")

# # (1) Leader model prediction
# leader_preds = aml.leader.predict(test_h2o).as_data_frame()["predict"]
# sub_leader = pd.DataFrame({
#     ID_COL: test[ID_COL],
#     LABEL: leader_preds
# })
# sub_leader.to_csv("submission_h2o_leader.csv", index=False)
# print("Saved: submission_h2o_leader.csv")

# # (2) Stacked Ensemble (BestOfFamily)
# ensemble_best_id = None
# for mid in aml.leaderboard["model_id"].as_data_frame().values.flatten():
#     if "StackedEnsemble_BestOfFamily" in mid:
#         ensemble_best_id = mid
#         break

# if ensemble_best_id:
#     ensemble_model = h2o.get_model(ensemble_best_id)
#     ensemble_preds = ensemble_model.predict(test_h2o).as_data_frame()["predict"]
#     sub_ensemble_best = pd.DataFrame({
#         ID_COL: test[ID_COL],
#         LABEL: ensemble_preds
#     })
#     sub_ensemble_best.to_csv("submission_h2o_ensemble_best.csv", index=False)
#     print("Saved: submission_h2o_ensemble_best.csv")
# else:
#     print("No StackedEnsemble_BestOfFamily found in leaderboard.")

# # (3) Stacked Ensemble (AllModels)
# ensemble_all_id = None
# for mid in aml.leaderboard["model_id"].as_data_frame().values.flatten():
#     if "StackedEnsemble_AllModels" in mid:
#         ensemble_all_id = mid
#         break

# if ensemble_all_id:
#     ensemble_model_all = h2o.get_model(ensemble_all_id)
#     ensemble_preds_all = ensemble_model_all.predict(test_h2o).as_data_frame()["predict"]
#     sub_ensemble_all = pd.DataFrame({
#         ID_COL: test[ID_COL],
#         LABEL: ensemble_preds_all
#     })
#     sub_ensemble_all.to_csv("submission_h2o_ensemble_all.csv", index=False)
#     print("Saved: submission_h2o_ensemble_all.csv")
# else:
#     print("No StackedEnsemble_AllModels found in leaderboard.")

# stack_h2o_lgbm_meta.py










"""
h2o_automl_train_full.py
- Full run (안정 버전)
- 데이터 전체 사용 + AutoML 30분 + LightGBM 튜닝
- Soft Voting / Stacking 포함
"""

import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# H2O
import h2o
from h2o.automl import H2OAutoML

# Sklearn / LGBM
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

# ================= Config =================
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
SUB_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")

LABEL = "target"
ID_COL = "ID"

# Full run 모드
H2O_MAX_RUNTIME_SECS = 1800   # 30분
H2O_MAX_MODELS       = 15
H2O_NFOLDS           = 5
H2O_EXCLUDE_ALGOS    = ["DeepLearning"]  # (XGBoost 안 깔려있으면 자동 skip)
H2O_BALANCE_CLASSES  = True
SEED = 42
N_SPLITS = 5

# Output folder
OUT_DIR = "submissions"
os.makedirs(OUT_DIR, exist_ok=True)

# ================= Utility =================
def save_sub(name, preds, ids, outdir=OUT_DIR):
    df = pd.DataFrame({ID_COL: ids, LABEL: preds})
    path = os.path.join(outdir, f"submission_{name}.csv")
    df.to_csv(path, index=False)
    print(f"  - Saved: {path}")
    return path

# ================= Load data =================
print("▶ Loading data ...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
sub   = pd.read_csv(SUB_CSV)

class_names = np.sort(train[LABEL].unique())
n_classes = len(class_names)
print(f"Train: {train.shape}, Test: {test.shape}, Classes: {n_classes}")

# ================= Preprocessing =================
num_cols = [c for c in train.columns if c not in [ID_COL, LABEL]]

# log transform
LOG_COLS = [c for c in ["X_11","X_19","X_37","X_40"] if c in train.columns]
for col in LOG_COLS:
    if (train[col] >= 0).all() and (test[col] >= 0).all():
        train[col] = np.log1p(train[col])
        test[col]  = np.log1p(test[col])

# scaling
scaler = RobustScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols]  = scaler.transform(test[num_cols])

# ================= StratifiedKFold =================
print("▶ Building stratified folds ...")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ================= H2O AutoML =================
print("▶ Starting H2O cluster ...")
h2o.init(max_mem_size="8G", nthreads=-1)

train_h2o = h2o.H2OFrame(train)
test_h2o  = h2o.H2OFrame(test)
train_h2o[LABEL] = train_h2o[LABEL].asfactor()

x = [c for c in train.columns if c not in [ID_COL, LABEL]]
y = LABEL

print("▶ Training H2O AutoML ...")
aml = H2OAutoML(
    max_runtime_secs=H2O_MAX_RUNTIME_SECS,
    max_models=H2O_MAX_MODELS,
    seed=SEED,
    nfolds=H2O_NFOLDS,
    balance_classes=H2O_BALANCE_CLASSES,
    exclude_algos=H2O_EXCLUDE_ALGOS,
    sort_metric="mean_per_class_error",
    stopping_metric="mean_per_class_error",
    stopping_tolerance=0.003,
    stopping_rounds=3,
    keep_cross_validation_predictions=True,
    keep_cross_validation_models=True,
    verbosity="info"
)
aml.train(x=x, y=y, training_frame=train_h2o)

leader = aml.leader
print("H2O Leader:", leader.model_id)

# H2O probs
h2o_test_pred_df = leader.predict(test_h2o).as_data_frame()
prob_cols_test = [c for c in h2o_test_pred_df.columns if c != "predict"]
h2o_test_probs = h2o_test_pred_df[prob_cols_test].to_numpy()
h2o_test_pred_labels = h2o_test_pred_df["predict"].to_numpy()

# ================= LightGBM =================
print("▶ Training LightGBM ...")
X = train[x].to_numpy()
y_raw = train[LABEL].to_numpy()
X_test = test[x].to_numpy()

cls_to_idx = {c:i for i,c in enumerate(class_names)}
idx_to_cls = {i:c for c,i in cls_to_idx.items()}
y = np.array([cls_to_idx[val] for val in y_raw])

oof_probs_lgb = np.zeros((len(train), n_classes), dtype=float)

base_lgb_params = dict(
    objective="multiclass",
    num_class=n_classes,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1e-2,
    reg_lambda=1e-2,
    random_state=SEED,
    n_estimators=300
)

# 튜닝
param_dist = {
    "num_leaves": [31, 63, 95, 127],
    "learning_rate": [0.01, 0.02, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1],
    "reg_lambda": [0.0, 1e-3, 1e-2, 1e-1],
    "min_child_samples": [10, 20, 40, 80]
}
tuner = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(**base_lgb_params),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_log_loss",
    cv=3,
    random_state=SEED,
    n_jobs=-1
)
tuner.fit(X, y)
lgb_params = {**base_lgb_params, **tuner.best_params_}

for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss")
    oof_probs_lgb[va_idx] = model.predict_proba(X_va)

lgb_full = lgb.LGBMClassifier(**lgb_params)
lgb_full.fit(X, y)
lgb_test_probs = lgb_full.predict_proba(X_test)

# ================= Weighted Soft Voting =================
print("▶ Weighted Soft Voting ...")
y_str = np.array(list(map(str, y_raw)))

h2o_oof = leader.cross_validation_holdout_predictions().as_data_frame()
prob_cols_oof = [c for c in h2o_oof.columns if c != "predict"]
h2o_oof_probs = h2o_oof[prob_cols_oof].to_numpy()

h2o_oof_pred_idx = h2o_oof_probs.argmax(axis=1)
h2o_pred_labels = np.array([prob_cols_oof[i] for i in h2o_oof_pred_idx])
h2o_oof_f1 = f1_score(y_str, h2o_pred_labels, average="macro")

lgb_oof_pred_idx = oof_probs_lgb.argmax(axis=1)
lgb_pred_labels = np.array([class_names[i] for i in lgb_oof_pred_idx])
lgb_oof_f1 = f1_score(y_str, lgb_pred_labels.astype(str), average="macro")

total = h2o_oof_f1 + lgb_oof_f1
w_h2o = h2o_oof_f1 / total if total > 1e-12 else 0.5
w_lgb = lgb_oof_f1 / total if total > 1e-12 else 0.5
print(f"  Weights -> H2O: {w_h2o:.3f}, LGBM: {w_lgb:.3f}")

soft_probs = w_h2o * h2o_test_probs + w_lgb * lgb_test_probs
soft_pred = np.array([class_names[i] for i in soft_probs.argmax(axis=1)])

# ================= Save submissions =================
print("▶ Saving submissions ...")
save_sub("h2o_leader", h2o_test_pred_labels, test[ID_COL].values)
save_sub("lgb_full", np.array([class_names[i] for i in lgb_full.predict(X_test)]), test[ID_COL].values)
save_sub("softvote_weighted", soft_pred, test[ID_COL].values)

# ================= Report =================
print(f"H2O OOF Macro-F1: {h2o_oof_f1:.5f}")
print(f"LGBM OOF Macro-F1: {lgb_oof_f1:.5f}")

# ================= Shutdown =================
try:
    h2o.cluster().shutdown()
except Exception:
    pass

print("▶ Done.")













# """
# h2o_automl_train_safe.py
# - Robustified pipeline that:
#   * Runs H2O AutoML (with fallback if AutoML produces no models)
#   * Trains LightGBM (OOF + full-train)
#   * Produces submission files (H2O leader if exists, LGB full, optional stacking/softvote)
#   * Quick test mode for fast development to avoid long waits
# """

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import time
# import numpy as np
# import pandas as pd

# # H2O
# import h2o
# from h2o.automl import H2OAutoML

# # Sklearn / LGBM
# from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import RobustScaler
# import lightgbm as lgb

# # ================= Config =================
# DATA_DIR = "data"
# TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
# TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
# SUB_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")

# LABEL = "target"
# ID_COL = "ID"

# # Quick mode => small sample, short budgets (use for debugging)
# QUICK_MODE = True

# if QUICK_MODE:
#     # Quick settings (aim ~ <=10 minutes dev run)
#     H2O_MAX_RUNTIME_SECS = 360     # 6 minutes for H2O (may still return empty if too small)
#     H2O_MAX_MODELS       = 2
#     H2O_NFOLDS           = 3
#     SAMPLE_FRAC = 0.2            # use 20% of train for quick runs
#     N_SPLITS = 3
# else:
#     H2O_MAX_RUNTIME_SECS = 1800   # 30 min
#     H2O_MAX_MODELS       = 15
#     H2O_NFOLDS           = 5
#     SAMPLE_FRAC = 1.0
#     N_SPLITS = 5

# H2O_EXCLUDE_ALGOS    = ["DeepLearning"]  # can leave empty if you want DL
# H2O_BALANCE_CLASSES  = True
# SEED = 42

# # LightGBM base params (objective, num_class 빼고)
# base_lgb_params = dict(
#     learning_rate=0.05 if not QUICK_MODE else 0.1,
#     max_depth=-1,
#     num_leaves=63 if not QUICK_MODE else 31,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_alpha=1e-2,
#     reg_lambda=1e-2,
#     random_state=SEED,
#     n_estimators=300 if not QUICK_MODE else 50
# )


# # Randomized tuning flag for LGB (turn off during quick)
# ENABLE_LGB_TUNING = False if QUICK_MODE else True
# LGB_TUNE_ITER = 25 if not QUICK_MODE else 5

# # Output folder
# OUT_DIR = "submissions"
# os.makedirs(OUT_DIR, exist_ok=True)

# # ============== Utility functions ==============
# def save_sub(name, preds, ids, outdir=OUT_DIR):
#     df = pd.DataFrame({ID_COL: ids, LABEL: preds})
#     path = os.path.join(outdir, f"submission_{name}.csv")
#     df.to_csv(path, index=False)
#     print(f"  - Saved: {path}")
#     return path

# def safe_as_str_array(arr):
#     """Convert array-like to numpy array of strings"""
#     return np.array(list(map(str, np.asarray(arr))))

# # ============== 0) Load data ==============
# print("▶ Loading data ...")
# train = pd.read_csv(TRAIN_CSV)
# test  = pd.read_csv(TEST_CSV)
# sub   = pd.read_csv(SUB_CSV)

# print(f"   Train shape: {train.shape}, Test shape: {test.shape}")

# # Optionally sample for quick debugging
# if SAMPLE_FRAC < 1.0:
#     print(f"   QUICK_MODE: sampling {SAMPLE_FRAC*100:.0f}% of training data for speed")
#     train = train.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
#     print(f"   After sampling train shape: {train.shape}")

# # ================== 1) Preprocessing ==================
# num_cols = [c for c in train.columns if c not in [ID_COL, LABEL]]
# # fill medians
# for c in num_cols:
#     med = train[c].median()
#     train[c] = train[c].fillna(med)
#     if c in test.columns:
#         test[c] = test[c].fillna(med)

# # optional log-col transform if present
# LOG_COLS = [c for c in ["X_11","X_19","X_37","X_40"] if c in train.columns]
# for c in LOG_COLS:
#     if (train[c] >= 0).all() and (test[c] >= 0).all():
#         train[c] = np.log1p(train[c])
#         test[c]  = np.log1p(test[c])

# # scaling
# scaler = RobustScaler()
# train[num_cols] = scaler.fit_transform(train[num_cols])
# test[num_cols]  = scaler.transform(test[num_cols])

# # classes
# class_names = np.sort(train[LABEL].unique())
# n_classes = len(class_names)
# print(f"   Classes: {n_classes}")

# # ensure labels consistent type: use strings for evaluation & mapping
# train[LABEL] = train[LABEL].astype(str)

# # ============== 2) CV split (skf) ==============
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# # ============== 3) H2O AutoML (try/except + fallback) ==============
# h2o_available = True
# h2o_leader_exists = False
# h2o_test_pred_labels = None
# h2o_test_probs = None
# h2o_oof_probs = None

# try:
#     print("▶ Starting H2O cluster ...")
#     # tune memory/threads per your machine
#     h2o.init(max_mem_size="4G", nthreads=-1)
#     train_h2o = h2o.H2OFrame(train)
#     test_h2o  = h2o.H2OFrame(test)
#     train_h2o[LABEL] = train_h2o[LABEL].asfactor()
#     x_cols = [c for c in train.columns if c not in [ID_COL, LABEL]]

#     print("▶ Running H2O AutoML ... (this may take a while)")
#     aml = H2OAutoML(
#         max_runtime_secs=H2O_MAX_RUNTIME_SECS,
#         max_models=H2O_MAX_MODELS,
#         seed=SEED,
#         nfolds=H2O_NFOLDS,
#         balance_classes=H2O_BALANCE_CLASSES,
#         exclude_algos=H2O_EXCLUDE_ALGOS,
#         sort_metric="mean_per_class_error",
#         keep_cross_validation_predictions=True,
#         keep_cross_validation_models=True,
#         verbosity="info"
#     )
#     aml.train(x=x_cols, y=LABEL, training_frame=train_h2o)

#     # check leaderboard
#     lb = aml.leaderboard if hasattr(aml, "leaderboard") else None
#     if lb is None or lb.nrows == 0:
#         print("⚠️ H2O AutoML produced no models (empty leaderboard). Will skip H2O outputs.")
#     else:
#         leader = aml.leader
#         if leader is None:
#             print("⚠️ No leader found in AutoML. Skipping H2O predictions.")
#         else:
#             h2o_leader_exists = True
#             print("H2O leader:", leader.model_id)

#             # get test preds (labels + probs if available)
#             h2o_test_pred_df = leader.predict(test_h2o).as_data_frame()
#             if "predict" in h2o_test_pred_df.columns:
#                 h2o_test_pred_labels = safe_as_str_array(h2o_test_pred_df["predict"].values)
#             else:
#                 h2o_test_pred_labels = None

#             # probability columns (may be absent for some models)
#             prob_cols = [c for c in h2o_test_pred_df.columns if c != "predict"]
#             if len(prob_cols) == n_classes:
#                 # reorder by class_names if possible (class_names are strings)
#                 ordered = [c for c in map(str, class_names) if c in prob_cols]
#                 if len(ordered) == n_classes:
#                     h2o_test_probs = h2o_test_pred_df[ordered].to_numpy()
#                 else:
#                     # fallback: use columns in reported order
#                     h2o_test_probs = h2o_test_pred_df[prob_cols].to_numpy()
#             else:
#                 h2o_test_probs = None

#             # OOF holdout preds (cross-validation holdout)
#             try:
#                 cv_holdout = leader.cross_validation_holdout_predictions().as_data_frame()
#                 pcols = [c for c in cv_holdout.columns if c != "predict"]
#                 if len(pcols) >= 1:
#                     # try to align with class_names, otherwise use what's present
#                     ordered_oof = [c for c in map(str, class_names) if c in pcols]
#                     if len(ordered_oof) == n_classes:
#                         h2o_oof_probs = cv_holdout[ordered_oof].to_numpy()
#                     else:
#                         h2o_oof_probs = cv_holdout[pcols].to_numpy()
#                 else:
#                     h2o_oof_probs = None
#             except Exception as e:
#                 print("   ⚠️ Couldn't get H2O cross_validation_holdout_predictions():", e)
#                 h2o_oof_probs = None

# except Exception as e:
#     print("⚠️ H2O AutoML step failed or skipped:", str(e))
#     h2o_available = False
# finally:
#     # If H2O started, we don't shut down yet — keep to inspect if needed.
#     pass

# # ============== 4) LightGBM OOF + Full-train ==============
# print("▶ Training LightGBM (OOF + full-train) ...")
# X = train[[c for c in train.columns if c not in [ID_COL, LABEL]]].to_numpy()
# y_str = train[LABEL].astype(str).to_numpy()  # 원라벨 문자열
# unique_classes = np.sort(np.unique(y_str))
# cls_to_idx = {c:i for i,c in enumerate(unique_classes)}
# idx_to_cls = {i:c for c,i in cls_to_idx.items()}
# y_idx = np.array([cls_to_idx[s] for s in y_str])

# X_test = test[[c for c in test.columns if c not in [ID_COL]]].to_numpy()
# n_classes = len(unique_classes)
# oof_probs_lgb = np.zeros((len(train), n_classes), dtype=float)

# # (옵션) 튜닝 후 안전장치
# if ENABLE_LGB_TUNING:
#     # ... (네가 쓰던 RandomizedSearchCV 그대로) ...
#     # 튜닝 결과 병합 직후 중복 키 제거
#     for k in ("objective", "num_class"):
#         lgb_params.pop(k, None)
# else:
#     lgb_params = dict(base_lgb_params)

# # CV OOF
# for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, y_idx)):
#     print(f"   LGB fold {fold_id+1}/{N_SPLITS} ...")
#     X_tr, X_va = X[tr_idx], X[va_idx]
#     y_tr, y_va = y_idx[tr_idx], y_idx[va_idx]

#     model = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **lgb_params)
#     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss")
#     oof_probs_lgb[va_idx] = model.predict_proba(X_va)

# # Full-train
# print("   Training LGB full model for test predictions ...")
# lgb_full = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **lgb_params)
# lgb_full.fit(X, y_idx)
# lgb_test_probs = lgb_full.predict_proba(X_test)

# # Weighted Soft Voting (H2O probs와 모양 맞을 때만)
# soft_pred_labels = None
# try:
#     if (h2o_test_probs is not None and
#         h2o_test_probs.shape == lgb_test_probs.shape):

#         # OOF F1들
#         h2o_oof_f1 = 0.0
#         if h2o_oof_probs is not None and h2o_oof_probs.shape[0] == len(train):
#             h2o_oof_pred_idx = h2o_oof_probs.argmax(axis=1)
#             h2o_oof_pred_str = np.array([idx_to_cls[i] for i in h2o_oof_pred_idx])
#             h2o_oof_f1 = f1_score(y_str, h2o_oof_pred_str, average="macro")

#         lgb_oof_pred_idx = oof_probs_lgb.argmax(axis=1)
#         lgb_oof_pred_str = np.array([idx_to_cls[i] for i in lgb_oof_pred_idx])
#         lgb_oof_f1 = f1_score(y_str, lgb_oof_pred_str, average="macro")

#         total = h2o_oof_f1 + lgb_oof_f1
#         w_h2o = 0.5 if total <= 1e-12 else h2o_oof_f1 / total
#         w_lgb = 0.5 if total <= 1e-12 else lgb_oof_f1 / total
#         print(f"▶ Weighted Soft Voting: H2O={w_h2o:.3f}, LGB={w_lgb:.3f}")

#         soft_probs = w_h2o * h2o_test_probs + w_lgb * lgb_test_probs
#         soft_idx = soft_probs.argmax(axis=1)
#         soft_pred_labels = np.array([idx_to_cls[i] for i in soft_idx])
#     else:
#         print("▶ Skipping soft vote: probs unavailable or shape mismatch")
# except Exception as e:
#     print("   ⚠️ Soft voting failed:", e)

# # 저장
# print("▶ Saving submissions ...")
# # H2O leader 라벨(있으면)
# if h2o_leader_exists and (h2o_test_pred_labels is not None):
#     save_sub("h2o_leader", h2o_test_pred_labels, test[ID_COL].values)

# # LGB full 라벨
# lgb_full_pred_idx = lgb_full.predict(X_test)
# lgb_full_pred_labels = np.array([idx_to_cls[i] for i in lgb_full_pred_idx])
# save_sub("lgb_full", lgb_full_pred_labels, test[ID_COL].values)

# # Soft vote 라벨(있으면)
# if soft_pred_labels is not None:
#     save_sub("soft_vote_weighted", soft_pred_labels, test[ID_COL].values)

# # OOF 리포트
# print("\n▶ Quick CV report:")
# print("  - LGB OOF Macro-F1:", f1_score(y_str, lgb_oof_pred_str, average="macro"))
# if h2o_oof_probs is not None and h2o_oof_probs.shape[0] == len(train):
#     print("  - H2O OOF Macro-F1:", f1_score(y_str, h2o_oof_pred_str, average="macro"))


# # ============== 9) Shutdown H2O (if running) ==============
# try:
#     if h2o_available:
#         h2o.cluster().shutdown()
#         print("▶ H2O cluster shutdown.")
# except Exception:
#     pass

# print("▶ Done.")
