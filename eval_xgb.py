import xgboost as xgb
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from datetime import datetime

import test
import logging
import pickle


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

DATA_PATH = '../public/'
MODELS_D = '../xgb-models/%d'


def eval_auc(i, pred, path):
    logging.info("eval %d" % i)
    data = pd.read_csv('../con/%03d.csv' % i, index_col='user_id')

    user = pd.read_csv('../public/user_create_time.csv', index_col='user_id')
    user2 = user['user_create_time'].str.split('-', 1, expand=True)
    user2 = user2.astype('int')
    for index, row in data.iterrows():
        user_row = user2.loc[index]
        data.at[index, 'created'] = (user_row[0] - 2016) * 12 + user_row[1]

    guess = {}
    for slot in range(0, 28):
        proba = pred(data, slot)

        guess['time_slot_%d' % slot] = proba

    ans = pd.read_csv(DATA_PATH + 'label-%03d.csv' % i, index_col='user_id')
    index = ans.index.tolist()
    guess_df = pd.DataFrame(guess, index=range(min(index), min(index) + len(index)))
    guess_df = guess_df[ans.columns.tolist()]

    auc = test.cal_auc(ans, i, guess_df)
    auc.to_csv(path + '%s-%03d.csv'
               % (datetime.now().isoformat(timespec='minutes'), i))

    return guess_df


def predict(data, slot):
    xgb = joblib.load(MODELS_D % slot)
    pred_xgb = xgb.predict_proba(data)[:, 1]

    # knn = pickle.load(open('../knn-models/%d' % slot, 'rb'))
    # knn = joblib.load('../knn-models/%d' % slot)
    # pred_knn = knn.predict_proba(data)[:, 1]

    # proba = 0.8 * pred_xgb + 0.2 * pred_knn

    # sgd = pickle.load(open('../sgd-models/%d' % slot, 'rb'))
    # pred_sgd = sgd.predict_proba(data)[:, 1]

    # proba = 0.9 * pred_xgb + 0.1 * pred_sgd
    proba = pred_xgb

    return proba


if __name__ == '__main__':
    for i in range(35, 46):
        eval_auc(i, predict, '../xgb-auc/')
