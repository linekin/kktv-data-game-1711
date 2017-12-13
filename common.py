import pickle

import pandas as pd
import logging
import random
import joblib

from sklearn.model_selection import cross_val_score, GridSearchCV

TIME_SLOT__D = 'time_slot_%d'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


def load_data(data_range=range(1, 41)):
    data_frames = []
    ans_frames = []
    for i in data_range:
        data_frames.append(pd.read_csv('../sess/%03d.csv' % i, index_col='user_id'))
        ans_frames.append(pd.read_csv('../public/label-%03d.csv' % i))

    data = pd.concat(data_frames)
    ans = pd.concat(ans_frames)
    return data, ans


def attempt(classifier, data_range=range(1, 41)):
    data, ans = load_data(data_range)

    scores = []
    for slot in random.sample(range(1, 28), 5):
        logging.info('cv %d' % slot)
        labels = ans[TIME_SLOT__D % slot]

        score = cross_val_score(classifier, data, labels, cv=4,
                                scoring='roc_auc',
                                n_jobs=-1)
        print(score)

        scores.append(score)

    return scores


def search(classifier, param_test, data_range=range(1, 41)):
    data, ans = load_data(data_range)

    gsearchs = []
    for slot in random.sample(range(1, 28), 5):
        logging.info('search %d' % slot)
        labels = ans[TIME_SLOT__D % slot]

        gsearch = GridSearchCV(
            classifier,
            param_test,
            scoring='roc_auc',
            n_jobs=-1,
            iid=False,
            cv=4
        )
        gsearch.fit(data, labels)

        print(gsearch.grid_scores_)

        gsearchs.append(gsearchs)

    return gsearchs


def train(classifier, model_dir, data_range=range(1,41)):
    data, ans = load_data(data_range)
    for slot in range(0, 28):
        logging.info('train %d' % slot)
        labels = ans[TIME_SLOT__D % slot]

        classifier.fit(data, labels)

        pickle.dump(classifier, open(model_dir + '%d' % slot, 'wb'))
        # joblib.dump(classifier, model_dir + '%d' % slot)
