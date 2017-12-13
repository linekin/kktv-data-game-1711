import pandas as pd
from auc import auc

DATA_PATH = '../public/'

i = 1

ans = pd.read_csv(DATA_PATH + 'label-%03d.csv' % i, index_col='user_id')

preds = pd.read_csv('../predict-%03d.csv' % i, index_col='user_id')
preds['auc'] = 0.0

for index, pred in preds.iterrows():
    truth = ans.loc[index]
    a = auc(pred[0:28], truth)
    preds.at[index, 'auc'] = round(a,4)

cols = preds.columns.tolist()
cols = cols[-1:] + cols[:-1]
preds = preds[cols]
preds.to_csv('../auc-%03d.csv' % i)
