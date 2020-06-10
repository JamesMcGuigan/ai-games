import os
from pathlib import Path

from src.settings import settings



mode     = 'test' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') else 'test'
mode_dir = mode
if   mode=='eval':  mode_dir = 'evaluation'
elif mode=='train': mode_dir = 'training'
elif mode=='test':  mode_dir = 'test'
else: raise Exception(f'invalid mode: {mode}')

data_path       = Path(settings['dir']['data'])
task_path       = data_path / mode
training_path   = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path       = data_path / 'test'
output_dir      = Path( settings['dir']['output'] )
