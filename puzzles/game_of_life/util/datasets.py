import os

import pandas as pd

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    directory = os.path.join(os.path.dirname(__file__), '../../input/conways-reverse-game-of-life-2020/')
else:
    directory = os.path.join(os.path.dirname(__file__), '../input/')

train_df  = pd.read_csv(f'{directory}/train.csv', index_col='id')
test_df   = pd.read_csv(f'{directory}/test.csv',  index_col='id')
