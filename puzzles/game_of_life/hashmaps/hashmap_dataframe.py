from joblib import delayed
from joblib import Parallel

from hashmaps.hash_functions import hash_geometric
from hashmaps.hash_functions import hash_translations
from hashmaps.hash_functions import max_index_x
from hashmaps.hash_functions import max_index_y
from utils.datasets import *
from utils.util import csv_column_names
from utils.util import csv_to_numpy_list


def hashmap_dataframe(df: pd.DataFrame, key='start'):
    boards                = csv_to_numpy_list(df, key=key)
    geometric_hashes      = Parallel(-1)( delayed(hash_geometric)(board)    for board in boards )
    translation_hashes    = Parallel(-1)( delayed(hash_translations)(board) for board in boards )
    max_x                 = Parallel(-1)( delayed(max_index_x)(board)       for board in boards )
    max_y                 = Parallel(-1)( delayed(max_index_y)(board)       for board in boards )

    output = df.copy(deep=True)
    output[f'{key}_geometric_hash']     = geometric_hashes
    output[f'{key}_translation_hashes'] = translation_hashes
    output[f'{key}_max_x']              = max_x
    output[f'{key}_max_y']              = max_y

    output = output.astype('int64')
    output = output.astype({ col: 'int8' for col in csv_column_names(key) })
    return output


if __name__ == '__main__':
    train_df = hashmap_dataframe(train_df, key='start')
    train_df = hashmap_dataframe(train_df, key='stop')
    test_df  = hashmap_dataframe(train_df, key='stop')

    train_df.to_csv('./output/hashmaps_train.csv')
    test_df.to_csv('./output/hashmaps_test.csv')
