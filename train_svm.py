from sklearn.svm import SVC

import common

common.train(
    SVC(),
    '../svm-models/'
)
