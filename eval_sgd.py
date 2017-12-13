import pickle

import eval_xgb


def predict(data, slot):
    classifier = pickle.load(open('../sgd-models/%d' % slot, 'rb'))
    return classifier.predict_proba(data)[:, 1]


for i in range(36, 46):
    eval_xgb.eval_auc(i, predict, '../sgd-auc/')
