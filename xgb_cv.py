import xgboost as xgb
import pandas as pd
import common
import random

params = {
    0: {
        'objective': 'binary:logistic',
        'nthread': 8,
        'tree_method': 'hist',
        # 'max_depth': 5,
        # 'learning_rate': 0.05
        # 'max_depth': 4,
    },
    1: {
        'objective': 'binary:logistic',
        'nthread': 8,
        'tree_method': 'hist',
        'subsample': 0.7,
        'colsample_bytree': 0.65,
        'alpha': 0.01,
        'max_depth': 6,
        'min_child_weight': 6,
        'gamma': 0.4,
        'eval_metrics': 'auc'
    }
}
param = 1

data, ans = common.load_data(range(1, 21))
# data = data.applymap(lambda x: 0 if x == 0 else 1)

user = pd.read_csv('../public/user_create_time.csv', index_col='user_id')
user2 = user['user_create_time'].str.split('-', 1, expand=True)
user2 = user2.astype('int')

for index, row in data.iterrows():
    user_row = user2.loc[index]
    data.at[index, 'created'] = (user_row[0] - 2016) * 12 + user_row[1]

features = xgb.DMatrix(data)

evals = []
# samples = random.sample(range(0, 28), 5)
# samples = [20,21,24,25]
samples = [20]
for slot in samples:
    labels = ans['time_slot_%d' % slot]
    features.set_label(labels)

    eval = xgb.cv(params[param],
                  features, 200,
                  nfold=5,
                  metrics='auc',
                  verbose_eval=True
                  )
    print(eval)
    evals.append(eval)
