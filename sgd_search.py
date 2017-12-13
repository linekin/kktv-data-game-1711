from sklearn.linear_model import SGDClassifier

import common

gsearches = common.search(
    SGDClassifier(
        loss='log',
        n_jobs=-1
    ), {
        'n_iter': [100, 120, 140, 160]
    }

)
