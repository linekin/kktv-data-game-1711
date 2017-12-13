import xgboost as xgb
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import common
import logging
import pickle
import os
import sys

X_train, y_train = common.load_data(range(1, 41))
X_test, y_test = common.load_data(range(41, 46))

user = pd.read_csv('../public/user_create_time.csv', index_col='user_id')
user2 = user['user_create_time'].str.split('-', 1, expand=True)
user2 = user2.astype('int')


def append_created(data):
    for index, row in data.iterrows():
        user_row = user2.loc[index]
        data.at[index, 'created'] = (user_row[0] - 2016) * 12 + user_row[1]


append_created(X_train)
append_created(X_test)

# X_train, X_test, y_train, y_test = train_test_split(
#     data, ans, test_size=0.1, random_state=0)

params = {
    'objective': 'binary:logistic',
    'nthread': 8,
    'tree_method': 'hist',
    'max_depth': 5,
    'n_estimators': 1000,
    'learning_rate': 0.05
}

MODEL_DIR = '../xgb-models/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
else:
    print('model dir already existed')
    sys.exit(1)

models = []
ranges = range(0, 28)
# ranges = [20,21,24,25]

for slot in ranges:
    logging.info('train %d' % slot)
    labels = y_train[common.TIME_SLOT__D % slot]

    xgb = XGBClassifier(**params)
    xgb.fit(
        X_train,
        labels,
        eval_set=[(X_test, y_test[common.TIME_SLOT__D % slot])],
        eval_metric='auc',
        early_stopping_rounds=100
    )
    models.append(xgb)

# train again for best iterations
for slot in ranges:
    logging.info('train %d' % slot)
    labels = y_train[common.TIME_SLOT__D % slot]

    params['n_estimators'] = models[slot].best_iteration
    xgb = XGBClassifier(**params)
    xgb.fit(
        X_train,
        labels,
        # eval_set=[(X_test, y_test[common.TIME_SLOT__D % slot])],
        # eval_metric='auc',
    )

    pickle.dump(xgb, open(MODEL_DIR + '%d' % slot, 'wb'))
