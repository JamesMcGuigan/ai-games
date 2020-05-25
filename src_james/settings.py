# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os
import pathlib
try:    root_dir = pathlib.Path(__file__).parent.parent.absolute()
except: root_dir = ''

if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') and 'submission' in __file__:
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Batch'

settings = {
    'verbose': True,
    'debug':   not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''),
    'caching': bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')) or False,
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/abstraction-and-reasoning-challenge/",
        "output":      "./",
    }
else:
    settings['dir'] = {
        "data":        os.path.join(root_dir, "./input"),
        "output":      os.path.join(root_dir, "./submission"),
    }

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    # for key,value in settings.items():  print(f"settings['{key}']:".ljust(30), str(value))
