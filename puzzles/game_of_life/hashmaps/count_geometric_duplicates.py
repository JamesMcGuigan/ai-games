# Kaggle Notebook: https://www.kaggle.com/jamesmcguigan/game-of-life-hashmap-solver/

from collections import defaultdict

from hashmaps.hashmap_dataframe import hashmap_test_df
from hashmaps.hashmap_dataframe import hashmap_train_df


def count_geometric_duplicates():
    # Create hashtable index for train_df
    train_stop_geometric_rows    = defaultdict(list)
    train_stop_translation_rows  = defaultdict(list)
    for idx, train_row in hashmap_train_df.iterrows():
        delta                 = train_row['delta']
        stop_geometric_hash   = train_row['stop_geometric_hash']
        stop_translation_hash = train_row['stop_translation_hash']
        train_stop_geometric_rows[stop_geometric_hash]     += [ train_row ]
        train_stop_translation_rows[stop_translation_hash] += [ train_row ]


    # Now count the number of hash matches in test_df
    count_exact       = 0
    count_geometric   = 0
    count_translation = 0
    count_total       = 0
    for idx, test_row in hashmap_test_df.iterrows():
        delta                      = test_row['delta']
        test_stop_geometric_hash   = test_row['stop_geometric_hash']
        test_stop_translation_hash = test_row['stop_translation_hash']

        count_total += 1

        # See if we find any geometric or translation hash matches
        if test_stop_translation_hash in train_stop_translation_rows:
            count_translation += 1

        if test_stop_geometric_hash in train_stop_geometric_rows:
            count_geometric += 1
            for train_row in train_stop_geometric_rows[test_stop_geometric_hash]:
                if train_row['delta'] == delta:
                    count_exact += 1
                    break

    print(" | ".join([
        f'count_exact = {count_exact} ({100*count_exact/count_total:.1f}%)',
        f'count_geometric = {count_geometric} ({100*count_geometric/count_total:.1f}%)',
        f'count_translation = {count_translation} ({100*count_translation/count_total:.1f}%)',
        f'count_total = {count_total}'
    ]))


if __name__ == '__main__':
    count_geometric_duplicates()
