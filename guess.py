import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


def get_slot(date):
    date_shift = date - timedelta(hours=1)
    day = date_shift.weekday()

    index = day_slot(date_shift)
    slot = day * 4 + index
    return slot


def day_slot(date_shift):
    if date_shift.hour < 8:
        index = 0
    elif date_shift.hour < 16:
        index = 1
    elif date_shift.hour < 20:
        index = 2
    else:
        index = 3
    return index


def process(filename, answer):
    logging.info('process ' + filename)
    data = pd.read_csv(filename)
    group_by_user = data.groupby('user_id')

    p = mp.Pool(8)

    name_groups = []
    for name, group in group_by_user:
        name_groups.append((answer, name, group))

    grouped_results = p.map(process_group, name_groups)
    return pd.concat(grouped_results)


AUG14 = datetime(2017, 8, 14, 1)


def process_group(name_groups):
    answer, name, group = name_groups
    for index, row in group.iterrows():
        event_time = row['event_time']
        date = parse(event_time)

        if date > AUG14:
            continue

        slot = get_slot(date)
        weight = date.month
        answer.at[name, 'time_slot_' + str(slot)] += weight
        # break
    return answer.loc[[name]]


if __name__ == '__main__':
    answer = pd.read_csv('../sample.csv', index_col='user_id')

    results = []
    for i in range(46, 76):
        results.append(process('../public/data-0%d.csv' % i, answer))

    pd.concat(results).to_csv('../ans%s.csv' % datetime.now().isoformat(timespec='minutes'))

# answer.to_csv('../ans%s.csv' % datetime.now().isoformat(timespec='minutes'))
