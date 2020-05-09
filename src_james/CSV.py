import os
import re
from typing import Union, List

import numpy as np

from src_james.settings import settings


class CSV:
    @staticmethod
    def write_submission(dataset: 'Dataset', filename='submission.csv'):
        csv        = CSV.to_csv(dataset)
        line_count = len(csv.split('\n'))
        filename   = os.path.join(settings['dir']['output'], filename)
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    @staticmethod
    def object_id(filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @staticmethod
    def to_csv(dataset: 'Dataset'):
        csv = ['output_id,output']
        for task in dataset:
            csv.append(CSV.to_csv_line(task))
        return "\n".join(csv)

    @staticmethod
    def to_csv_line(task: 'Task') -> str:
        csv = []
        for i, problem in enumerate(task['test']):
            line = ",".join([
                CSV.object_id(task.filename, i),
                CSV.problem_to_csv_string(problem)
            ])
            csv.append(line)
        return "\n".join(csv)

    @staticmethod
    def problem_to_csv_string(problem: 'Problem') -> str:
        # TODO: Do we need to consider a range of possible solutions?
        if problem['solution']: return CSV.grid_to_csv_string( problem.data['solution'] )
        else:                   return CSV.grid_to_csv_string( problem.data['input']    )

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
