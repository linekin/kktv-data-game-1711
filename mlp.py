from sklearn.neural_network import MLPClassifier

import common

scores = common.attempt(
    MLPClassifier(),
    range(1, 3))
