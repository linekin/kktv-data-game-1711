import sys
import pandas as pd


def normalize(ans):
    for i in range(0, 28):
        name = 'time_slot_%d' % i
        ans.__setattr__(name, ans.__getattr__(name).astype(float))
    counter = 0
    for index, row in ans.iterrows():
        counter += 1
        if counter % 10000 == 0:
            print('processed %d' % counter)

        dividend = float(row.max()) * 8
        if dividend != 0:
            for c in row.index:
                v = ans.at[index, c]
                v /= dividend
                ans.at[index, c] = round(v, 4)


if __name__ == '__main__':
    ans = pd.read_csv(sys.argv[1], index_col='user_id')
    normalize(ans)
    ans.to_csv(sys.argv[2])
