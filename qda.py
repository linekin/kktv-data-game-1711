from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import common

scores = common.attempt(
    QuadraticDiscriminantAnalysis(),
    range(1, 3))
