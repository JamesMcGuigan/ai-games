# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

settings = {}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/abstraction-and-reasoning-challenge/",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input",
        "logs":        "./logs",
    }

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    # for key,value in settings.items():  print(f"settings['{key}']:".ljust(30), str(value))
