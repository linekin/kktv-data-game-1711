import pandas as pd
from dateutil.parser import parse
import numpy as np
from datetime import datetime, timedelta
import guess
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

DATA_PATH = '../public/'
SLOT_PER_DAY = 4
AUG14 = datetime(2017, 8, 14)


def consolidate(i):
    logging.info('consolidating %d' % i)
    raw_data = pd.read_csv(DATA_PATH + 'data-%03d.csv' % i)
    group_by_user = raw_data.groupby('user_id')
    users = group_by_user.groups.keys()
    user_min = min(users)
    aug13 = datetime(2017, 8, 13)
    aug13_yday = aug13.timetuple().tm_yday
    matrix = np.zeros((len(users), aug13_yday * SLOT_PER_DAY), np.uint32)
    data = pd.DataFrame(matrix, index=range(user_min, user_min + len(users)))

    name_groups = []
    # grouped_row = []
    for name, group in group_by_user:
        name_groups.append((data, name, group))
        # grouped_row.append(process_group((data, name, group)))

    p = mp.Pool(8)
    grouped_row = p.map(process_group, name_groups)

    pd.concat(grouped_row).to_csv('../con/%03d.csv' % i, index_label='user_id')


def process_group(name_groups):
    data, name, group = name_groups
    for index, row in group.iterrows():
        event_time = row['event_time']
        date = parse(event_time) - timedelta(hours=1)

        if date.year < 2017:
            continue

        if date > AUG14:
            continue

        yday = date.timetuple().tm_yday
        day_slot = guess.day_slot(date)
        slot = (yday - 1) * SLOT_PER_DAY + day_slot

        data.loc[name][slot] += 1

    return data.loc[[name]]


if __name__ == '__main__':
    for i in range(19, 46):
        consolidate(i)
