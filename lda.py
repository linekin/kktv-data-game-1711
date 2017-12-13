from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import common

scores = common.attempt(
    LinearDiscriminantAnalysis(),
    range(1,3)
)