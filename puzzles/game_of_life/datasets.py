import os

import pandas as pd

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    train_df  = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv', index_col='id')
    test_df   = pd.read_csv('../input/conways-reverse-game-of-life-2020/test.csv',  index_col='id')
else:
    train_df  = pd.read_csv('./input/train.csv', index_col='id')
    test_df   = pd.read_csv('./input/test.csv',  index_col='id')
