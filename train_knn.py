from sklearn.neighbors import KNeighborsClassifier

import common

common.train(
    KNeighborsClassifier(n_jobs=-1),
    '../knn-models/'
)
