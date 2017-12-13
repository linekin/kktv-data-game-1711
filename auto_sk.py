import autosklearn.classification
import pandas as pd
import pickle
import common
import logging


data_frames = []
ans_frames = []
for i in range(1, 41):
    data_frames.append(pd.read_csv('../con/%03d.csv' % i, index_col='user_id'))
    ans_frames.append(pd.read_csv('../public/label-%03d.csv' % i))
data = pd.concat(data_frames)
ans = pd.concat(ans_frames)

for slot in range(0, 28):
    logging.info("train %d" % slot)
    labels = ans['time_slot_%d' % slot]
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(data, labels)
    logging.info("save %d" % slot)
    pickle.dump(automl, open('../auto-models/%d' % slot, 'wb'))

