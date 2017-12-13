from sklearn.linear_model import SGDClassifier

import common

scores = common.attempt(
    SGDClassifier(
        n_jobs=-1
    ),
    range(1,3))