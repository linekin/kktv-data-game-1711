import common
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
from dateutil.parser import parse
import multiprocessing as mp

import guess

SLOT_PER_DAY = 4


def session_feature(i):
    logging.info('session feature #%d' % i)
    raw_data = pd.read_csv('../public/data-%03d.csv' % i)

    group_by_user = raw_data.groupby('user_id')
    users = group_by_user.groups.keys()
    user_min = min(users)
    aug13 = datetime(2017, 8, 13)
    aug13_yday = aug13.timetuple().tm_yday
    matrix = np.zeros((len(users), aug13_yday * SLOT_PER_DAY), np.uint32)
    data = pd.DataFrame(matrix, index=range(user_min, user_min + len(users)))

    name_groups = []
    for user_id, group in group_by_user:
        name_groups.append((data, user_id, group))

    p = mp.Pool(8)
    grouped_row = p.map(process_group, name_groups)
    pd.concat(grouped_row).to_csv(
        '../sess/%03d.csv' % i,
        index_label='user_id')


AUG14 = datetime(2017, 8, 14)


def process_group(name_groups):
    data, user_id, group = name_groups
    group_by_session = group.groupby('session_id')

    for session_id, sess_group in group_by_session:
        first_row = sess_group.iloc[0]
        last_row = sess_group.iloc[-1]

        first_slot = slot_of(first_row, True)
        last_slot = slot_of(last_row, False)

        if first_slot == -1 or last_slot == -1:
            continue

        for slot in range(first_slot, last_slot + 1):
            data.loc[user_id][slot] = 1

    return data.loc[[user_id]]


def slot_of(row, start):
    event_time = row['event_time']
    date = parse(event_time) - timedelta(hours=1)

    if start:
        date = date - timedelta(seconds=int(row['played_duration']))

    if date.year < 2017:
        return -1

    if date > AUG14:
        return -1

    yday = date.timetuple().tm_yday
    day_slot = guess.day_slot(date)
    slot = (yday - 1) * SLOT_PER_DAY + day_slot

    return slot


if __name__ == '__main__':
    for i in range(1, 76):
        session_feature(i)
