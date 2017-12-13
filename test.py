import pandas as pd
import sys
import guess
import normalize
from datetime import datetime
from auc import auc
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

DATA_PATH = '../public/'
PREFIX=datetime.now().isoformat(timespec='minutes')


def guess_validate(i):
    ans = pd.read_csv(DATA_PATH + 'label-%03d.csv' % i, index_col='user_id')
    ans0 = ans * 0

    preds = guess.process(DATA_PATH + 'data-%03d.csv' % i, ans0)
    # post-processing
    normalize.normalize(preds)
    # dt = datetime.now().isoformat(timespec='minutes')

    preds.to_csv('../predict/%s-%03d.csv' % (PREFIX, i))
    # predicting

    cal_auc(ans, i, preds)


def cal_auc(ans, i, preds):
    preds['auc'] = 0.0
    for index, pred in preds.iterrows():
        truth = ans.loc[index]
        a = auc(pred[0:28], truth)
        preds.at[index, 'auc'] = round(a, 4)
    cols = preds.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    preds = preds[cols]
    # preds.to_csv('../auc/%s-%03d.csv' % (PREFIX, i))
    return preds


if __name__ == '__main__':
    # i = sys.argv[1]

    dt = datetime.now().isoformat(timespec='minutes')
    # for i in random.sample(range(1, 46), 5):
    for i in [2, 21, 26, 42, 43]:
        guess_validate(i, dt)
