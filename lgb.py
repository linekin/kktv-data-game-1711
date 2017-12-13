import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import common


data, ans = common.load_data(range(1, 21))

estimator = lgb.LGBMClassifier(silent=False)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbms=[]
for slot in [0]:
    labels=ans['time_slot_%d' % slot]

    gbm = GridSearchCV(
        estimator, param_grid,
        scoring='roc_auc',
        n_jobs=4,
        iid=False)

    gbm.fit(data, labels)
    gbms.append(gbm)

    print('Best parameters found by grid search are:', gbm.best_params_)