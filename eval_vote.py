import pandas as pd

i = 45

data = pd.read_csv('../con/%03d.csv' % i,
                   index_col='user_id')
