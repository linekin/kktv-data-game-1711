import pandas as pd
import xgboost as xgb
import pickle
import joblib

SLOT_FMT = 'time_slot_%d'

ans = pd.read_csv('../sample.csv', index_col='user_id')

for i in range(0, 28):
    name = SLOT_FMT % i
    ans.__setattr__(name, ans.__getattr__(name).astype(float))

feature_frames = []
for i in range(46, 76):
    data = pd.read_csv('../sess/%03d.csv' % i, index_col='user_id')
    feature_frames.append(data)
features = pd.concat(feature_frames)

user = pd.read_csv('../public/user_create_time.csv', index_col='user_id')
user2 = user['user_create_time'].str.split('-', 1, expand=True)
user2 = user2.astype('int')
for index, row in features.iterrows():
    user_row = user2.loc[index]
    features.at[index, 'created'] = (user_row[0] - 2016) * 12 + user_row[1]


guess = {}
for slot in range(0, 28):
    xgb = pickle.load(open('../xgb-models/%d' % slot, 'rb'))
    pred_xgb = xgb.predict_proba(features)[:, 1]

    # sgd = pickle.load(open('../sgd-models/%d' % slot, 'rb'))
    # pred_sgd = sgd.predict_proba(features)[:, 1]

    # proba = 0.9 * pred_xgb + 0.1 * pred_sgd
    guess[SLOT_FMT % slot] = pred_xgb

for i in range(0, len(features)):
    for slot in range(0, 28):
        ans.iat[i, slot] = guess[SLOT_FMT % slot][i]
