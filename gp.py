from sklearn.gaussian_process import GaussianProcessClassifier
import common
import pickle
import logging
from sklearn.model_selection import cross_val_score

data, ans = common.load_data(range(1, 11))

classfiers = []
scores = []
for slot in range(1, 28):
    logging.info('cv %d' % slot)
    labels = ans['time_slot_%d' % slot]

    classifier = GaussianProcessClassifier()
    score = cross_val_score(classifier, data, labels, cv=5,
                            scoring='roc_auc')
    print(score)

    classfiers.append(classifier)
    scores.append(score)
    pickle.dump(classifier, open('../gp-models/%d' % slot, 'wb'))
