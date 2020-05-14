import os
import re
from typing import List, Union

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
            csv.append(CSV.to_csv_line(task))
        return "\n".join(csv)

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        for test_index, problem in enumerate(task['test']):
            if not problem.get('solution', None): continue
            solutions = {
                cls.grid_to_csv_string(solution)
                for solution in problem['solution']
            }
            for sol_index, solution_csv in enumerate(solutions):
                line = ",".join([
                    cls.object_id(task.filename, test_index+sol_index),
                    solution_csv
                ])
                csv.append(line)
        return "\n".join(csv)

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    @staticmethod
    def grid_to_csv_string(grid: Union[List[List[int]], np.ndarray]) -> str:
        if isinstance(grid, np.ndarray):
            grid = grid.astype('int8').tolist()
        str_pred = str([ row for row in grid ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred
