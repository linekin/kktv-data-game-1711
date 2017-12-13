import pickle

import eval_xgb


def predict(data, slot):
    knn = pickle.load(open('../knn-models/%d' % slot, 'rb'))
    return knn.predict_proba(data)[:, 1]


for i in range(36, 46):
    eval_xgb.eval_auc(i, predict, '../knn-auc/')
