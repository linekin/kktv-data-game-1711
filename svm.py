from sklearn.svm import SVC

import common

scores = common.attempt(SVC(kernel="linear"), range(1, 3))
