import json
import re
from glob import glob

import numpy as np



def load_tasks(task_files):
    if isinstance(task_files, str): task_files = glob(task_files)

    tasks = { re.sub(r'^.*/(.*?/.*$)','\\1',file): json.load(open(file, 'r')) for file in task_files }

    for file, task in tasks.items():
        for test_train, specs in task.items():
            for index, spec in enumerate(specs):
                for input_output, grid in spec.items():
                    tasks[file][test_train][index][input_output] = np.array(grid).astype('uint8')  # uint8 required for cv2

    for file, task in tasks.items():
        tasks[file]['file'] = file
    return tasks


def score_tasks(tasks: dict, label: str):
    total  = 0
    solved = 0

    for task_file, task in tasks.items():
        for index, spec in enumerate(task['test']): total += 1

        if 'solution' in task:
            for index, spec in enumerate(task['solution']):
                solved += 1

    print(f'{label.ljust(11)}| solutions found: {str(solved).rjust(3)}/{total} | error: {round(1-solved/total,4)}')
    return solved
