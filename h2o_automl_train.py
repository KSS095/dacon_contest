# h2o_automl_train.py

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# ================================
# 0) 데이터 로드
# ================================
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
sub   = pd.read_csv("data/sample_submission.csv")

LABEL = "target"
ID_COL = "ID"

# ================================
# 1) 전처리 (로그 변환 + 스케일링)
# ================================
# 로그 변환할 컬럼 (왜도 높은 컬럼 위주)
LOG_COLS = ["X_11", "X_19", "X_37", "X_40"]
for col in LOG_COLS:
    if col in train.columns:
        train[col] = np.log1p(train[col])
        test[col]  = np.log1p(test[col])

# RobustScaler (이상치에 강건)
scaler = RobustScaler()
X_cols = [c for c in train.columns if c not in [ID_COL, LABEL]]
train[X_cols] = scaler.fit_transform(train[X_cols])
test[X_cols]  = scaler.transform(test[X_cols])

# ================================
# 2) H2O 초기화
# ================================
h2o.init(max_mem_size="8G")   # 필요 시 조절 가능

# Pandas → H2OFrame 변환
train_h2o = h2o.H2OFrame(train)
test_h2o  = h2o.H2OFrame(test)

x = [c for c in train.columns if c not in [ID_COL, LABEL]]
y = LABEL

train_h2o[y] = train_h2o[y].asfactor()   # 분류 문제라서 factor 변환

# ================================
# 3) Feature Importance 기반 컬럼 선택 (GBM 빠르게 학습)
# ================================
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, seed=42)
gbm.train(x=x, y=y, training_frame=train_h2o)

# 중요도 확인
importance = gbm.varimp(use_pandas=True)
selected_features = importance[importance["relative_importance"] > 0.001]["variable"].tolist()

print(f"선택된 주요 Feature 개수: {len(selected_features)} / {len(x)}")

# ================================
# 4) AutoML 실행 (1시간 제한)
# ================================
aml = H2OAutoML(
    max_runtime_secs=3600,   # 실행 시간 제한 (1시간)
    max_models=20,           # 최대 모델 수 제한
    seed=42,
    nfolds=5,
    exclude_algos=["DeepLearning"],  # 느린 NN 제외 (원하면 주석 처리)
    sort_metric="mean_per_class_error"
)

aml.train(x=selected_features, y=y, training_frame=train_h2o)

# ================================
# 5) 결과 확인 및 제출 파일 생성
# ================================
lb = aml.leaderboard
print(lb.head(rows=10))

preds = aml.leader.predict(test_h2o)
sub_out = pd.DataFrame({ID_COL: test[ID_COL], LABEL: preds.as_data_frame().values.flatten()})

sub_out.to_csv("submission_h2o.csv", index=False)
print("Saved: submission_h2o.csv")
