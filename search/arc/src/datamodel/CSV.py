import os
import re

import numpy as np
import pydash

from src.settings import settings



# noinspection PyUnresolvedReferences
class CSV:
    @classmethod
    def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
        csv        = CSV.to_csv(dataset)
        line_count = len(csv.split('\n'))
        filename   = os.path.join(settings['dir']['output'], filename)
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    ### No need to extend sample_submission.csv, just sort the CSV
    # @classmethod
    # def sample_submission(cls):
    #     filename = os.path.join(settings['dir']['data'],'sample_submission.csv')
    #     sample_submission = pd.read_csv(filename)
    #     return sample_submission
    #
    # @classmethod
    # def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
    #     csv        = CSV.to_csv(dataset)
    #     lines      = csv.split('\n')
    #     line_count = len(lines)
    #     data       = []
    #
    #     submission = cls.sample_submission()
    #     submission = submission.set_index('output_id', drop=False)
    #     for line in lines[1:]:  # ignore header
    #         object_id,output = line.split(',',2)
    #         submission.loc[object_id]['output'] = output
    #
    #     submission.to_csv(filename, index=False)
    #     print(f"\nwrote: {filename} | {line_count} lines")


    @classmethod
    def object_id(cls, filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @classmethod
    def to_csv(cls, dataset: 'Dataset'):
        csv = []
        for task in dataset:
            line = CSV.to_csv_line(task)
            if line: csv.append(line)
        csv = ['output_id,output'] + sorted(csv) # object_id keys are sorted in sample_submission.csv
        return "\n".join(csv)

    # noinspection PyUnusedLocal
    @classmethod
    def default_csv_line(cls, task: 'Task' = None) -> str:
        return '|123|456|789|'

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        for index, problemset in enumerate(task['solutions']):
            solutions = pydash.uniq(list(
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
