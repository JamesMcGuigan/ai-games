# Kaggle Notebook: https://www.kaggle.com/jamesmcguigan/game-of-life-hashmap-solver/

from joblib import delayed
from joblib import Parallel

from hashmaps.hash_functions import hash_geometric
from hashmaps.hash_functions import hash_translations
from utils.datasets import *
from utils.util import csv_column_names
from utils.util import csv_to_numpy_list


def hashmap_dataframe(df: pd.DataFrame, key='start'):
    boards                = csv_to_numpy_list(df, key=key)
    geometric_hashes      = Parallel(-1)( delayed(hash_geometric)(board)    for board in boards )
    translation_hashes    = Parallel(-1)( delayed(hash_translations)(board) for board in boards )

    output = df.copy(deep=True)
    output[f'{key}_geometric_hash']   = geometric_hashes
    output[f'{key}_translation_hash'] = translation_hashes

    output = output.astype('int64')
    output = output.astype({ col: 'int8' for col in csv_column_names(key) })
    return output


hashmap_train_filename = f'{output_directory}/hashmap_train.csv'
hashmap_test_filename  = f'{output_directory}/hashmap_test.csv'
if os.path.exists(hashmap_train_filename) and os.path.exists(hashmap_test_filename):
    hashmap_train_df = pd.read_csv(hashmap_train_filename)
    hashmap_test_df  = pd.read_csv(hashmap_test_filename)
else:
    hashmap_train_df = train_df
    hashmap_test_df  = test_df
    hashmap_train_df = hashmap_dataframe(hashmap_train_df, key='start')
    hashmap_train_df = hashmap_dataframe(hashmap_train_df, key='stop')
    hashmap_test_df  = hashmap_dataframe(hashmap_test_df,  key='stop')

    hashmap_train_df.to_csv(hashmap_train_filename)
    hashmap_test_df.to_csv(hashmap_test_filename)
