from xgboost import XGBClassifier

import common

scores = common.attempt(
    XGBClassifier(
        objective='binary:logistic',
        n_estimators=18,
        n_jobs=8,
        subsample=0.7,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        max_depth=6,
        min_child_weight=5,
        gamma=0.4
    ),
    range(1,3)
)