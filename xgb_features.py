import operator

import xgboost as xgb

from datetime import datetime, timedelta, date

bst = xgb.Booster({'nthread': 4})

slot = 0
bst.load_model('../models/%d' % slot)
scores = bst.get_fscore()

# for slot in range(0,28):
sorted_x = sorted(
    scores.items(),
    key=operator.itemgetter(1),
    reverse=True)

for f, v in sorted_x:
    d = int(f[1:])
    days = int(d/4)
    feature_date = date(2017, 1, 1) + timedelta(days)

    slot = d - days * 4
    slot_desc  = {
        0: '1-9',
        1: '9-17',
        2: '17-21',
        3: '21-1'
    }
    print(feature_date, feature_date.weekday(), slot_desc[slot], v)