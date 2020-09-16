# $ kaggle c download conways-reverse-game-of-life-2020
# $ unzip conways-reverse-game-of-life-2020.zip -d input

import os

import pandas as pd

root_directory = os.path.join(os.path.dirname(__file__), '../')
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    input_directory  = os.path.join(root_directory, '../input/conways-reverse-game-of-life-2020/')
    output_directory = os.path.join(root_directory, './')
else:
    input_directory  = os.path.join(root_directory, './input/')
    output_directory = os.path.join(root_directory, './output/')


train_file             = f'{input_directory}/train.csv'
test_file              = f'{input_directory}/test.csv'
sample_submission_file = f'{input_directory}/sample_submission.csv'
submission_file        = f'{output_directory}/submission.csv'


assert os.path.exists(train_file)
assert os.path.exists(test_file)
assert os.path.exists(sample_submission_file)


# Ensure the submission.csv file exists
def copy_sample_submission_file():
    if not os.path.exists(submission_file) or os.path.getsize(submission_file) == 0:
        if os.path.dirname(submission_file): os.makedirs(os.path.dirname(submission_file), exist_ok=True)
        with open(sample_submission_file, 'r') as source, open(submission_file, 'w') as dest:
            print(f'copy_sample_submission_file(): {submission_file} missing, copying from {sample_submission_file}')
            dest.write(source.read())
copy_sample_submission_file()


train_df             = pd.read_csv(train_file, index_col='id')
test_df              = pd.read_csv(test_file,  index_col='id')
submission_df        = pd.read_csv(submission_file,  index_col='id')
sample_submission_df = pd.read_csv(sample_submission_file,  index_col='id')


# Ensure submission.csv contains all required indices and is in sorted order
def copy_sample_submission_missing_indexes():
    global submission_df
    idxs          = set(sample_submission_df.index) - set(submission_df.index)
    submission_df = pd.concat([ submission_df, sample_submission_df.loc[idxs] ])
    if len(idxs):
        print(f'copy_sample_submission_missing_indexes(): copied {len(idxs)} missing indexes from {sample_submission_file} to {submission_file}')
        submission_df.sort_index().to_csv(submission_file)
copy_sample_submission_missing_indexes()