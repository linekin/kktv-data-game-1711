from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import common
import random

gsearchs = common.search(
    SVC(),
    {
        'degree': [2, 3]
    }, range(1, 11))
