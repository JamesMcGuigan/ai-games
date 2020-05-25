import os
import re

import numpy as np

from src_james.settings import settings


class CSV:
    @classmethod
    def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
        csv        = CSV.to_csv(dataset)
        line_count = len(csv.split('\n'))
        filename   = os.path.join(settings['dir']['output'], filename)
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    @classmethod
    def object_id(cls, filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @classmethod
    def to_csv(cls, dataset: 'Dataset'):
        csv = ['output_id,output']
        for task in dataset:
            line = CSV.to_csv_line(task)
            if line: csv.append(line)
        return "\n".join(csv)

    @classmethod
    def default_csv_line(cls, task: 'Task') -> str:
        return '|123|456|789|'

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        for index, problemset in enumerate(task['solutions']):
            solutions = list(set(
                cls.grid_to_csv_string(problem['output'])
                for problem in problemset
            ))
            solution_str = " ".join(solutions[:3]) if len(solutions) else cls.default_csv_line(task)
            line = ",".join([
                cls.object_id(task.filename, index),
                solution_str
            ])
            csv.append(line)
        return "\n".join(csv)

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    # noinspection PyTypeChecker
    @staticmethod
    def grid_to_csv_string(grid: np.ndarray) -> str:
        if grid is None: return None
        grid = np.array(grid).astype('int8').tolist()
        str_pred = str([ row for row in grid ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred
