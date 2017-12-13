from sklearn.linear_model import SGDClassifier

import common

common.train(
    SGDClassifier(
        n_jobs=-1,
        loss='log',
        n_iter=100
    ),
    '../sgd-models/'
)
