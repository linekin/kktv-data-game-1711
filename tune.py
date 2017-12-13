import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import random
import common
import logging

classifier = XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    n_jobs=8,
    subsample=0.7,
    colsample_bytree=0.65,
    reg_alpha=0.1,
    max_depth=6,
    min_child_weight=5,
    scale_pos_weight=1,
    gamma=0.4,
    eval_metric='auc',
)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gsearch = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    # cv=5
)

data, ans = common.load_data(range(1, 21))

gsearchs = []
# ranges = random.sample(range(0, 28), 5)
# ranges=[20,21,24,25]
ranges=[0]
for slot in ranges:
    logging.info('search %d' % slot)
    labels = ans['time_slot_%d' % slot]
    gsearch.fit(data, labels)
    print(gsearch.grid_scores_)
    gsearchs.append(gsearch)
