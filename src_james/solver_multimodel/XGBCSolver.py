import os
import time

import numpy as np
from fastcache._lrucache import clru_cache
from xgboost import XGBClassifier

from src_james.core.DataModel import Task, Competition
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_unchanged
from src_james.util.np_cache import np_cache


class XGBSolver(Solver):
    optimise = True
    verbose  = False

    def __init__(self, local_neighb=5):
        self.local_neighb = local_neighb
        super().__init__()

    def detect(self, task):
        if not is_task_shape_ratio_unchanged(task): return False
        # inputs, outputs, not_valid = self.features(task)
        # if not_valid: return False
        return True

    def fit(self, task):
        if task.filename not in self.cache:
            # inputs  = task['train'].inputs + task['test'].inputs
            # outputs = task['train'].outputs
            inputs, outputs, not_valid = self.features(task)
            if not_valid:
                self.cache[task.filename] = None
            else:
                xgb = XGBClassifier(n_estimators=10, n_jobs=-1)
                xgb.fit(inputs, outputs, verbose=False)
                self.cache[task.filename] = (xgb, self.local_neighb)

    def test(self, task: Task) -> bool:
        """test if the given action correctly solves the task"""
        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)

    def action(self, grid, xgb=None, local_neighb=5, task=None):
        if task and task.filename not in self.cache: self.fit(task)
        xgb      = xgb or self.cache[task.filename][0]
        features = self.make_features(grid, local_neighb=local_neighb)
        predict  = xgb.predict(features)
        output   = predict.reshape(*grid.shape)
        return output


    @classmethod
    @clru_cache(None)
    def features(cls, task, mode='train'):
        num_train_pairs = len(task[mode])
        feat, target = [], []

        for task_num in range(num_train_pairs):
            input_color              = np.array(task[mode][task_num]['input'])
            target_color             = task[mode][task_num]['output']
            nrows, ncols             = len(task[mode][task_num]['input']),  len(task[mode][task_num]['input'][0])
            target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

            if (target_rows != nrows) or (target_cols != ncols):
                # print('Number of input rows:', nrows, 'cols:', ncols)
                # print('Number of target rows:', target_rows, 'cols:', target_cols)
                not_valid = 1
                return None, None, 1

            imsize = nrows * ncols
            # offset = imsize*task_num*3 #since we are using three types of aug
            feat.extend(cls.make_features(input_color))
            target.extend(np.array(target_color).reshape(-1, ))

        return np.array(feat), np.array(target), 0

    @classmethod
    @np_cache()
    def make_features(cls, input_color: np.ndarray, local_neighb=5):
        nrows, ncols = input_color.shape
        feat = np.zeros((nrows * ncols, 13))
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0]   = i
                feat[cur_idx, 1]   = j
                feat[cur_idx, 2]   = input_color[i][j]
                feat[cur_idx, 3:7] = cls.get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = cls.get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx, 9]   = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10]  = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11]  = (i + j)
                feat[cur_idx, 12]  = len(np.unique(
                    input_color[i - local_neighb:i + local_neighb,
                    j - local_neighb:j + local_neighb])
                )
                cur_idx += 1
        return feat

    @classmethod
    @np_cache()
    def get_moore_neighbours(cls, color, cur_row, cur_col, nrows, ncols):
        top    = -1 if cur_row <= 0         else color[cur_row - 1][cur_col]
        bottom = -1 if cur_row >= nrows - 1 else color[cur_row + 1][cur_col]
        left   = -1 if cur_col <= 0         else color[cur_row][cur_col - 1]
        right  = -1 if cur_col >= ncols - 1 else color[cur_row][cur_col + 1]
        return top, bottom, left, right


    @classmethod
    @np_cache()
    def get_tl_tr(cls, color, cur_row, cur_col, nrows, ncols):
        top_left  = -1 if cur_row == 0 or cur_col == 0         else color[cur_row - 1][cur_col - 1]
        top_right = -1 if cur_row == 0 or cur_col == ncols - 1 else color[cur_row - 1][cur_col + 1]
        return top_left, top_right


if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    competition = Competition()
    competition.time_start = time.perf_counter()
    solver = XGBSolver()
    for name, dataset in competition.items():
        solver.solve_all(dataset)
    competition.time_taken = time.perf_counter() - competition.time_start
    print(competition)

    # for local_neighb in [1,30]:
    #     print(f'local_neighb: {local_neighb}')
    #     competition.time_start = time.perf_counter()
    #     solver = XGBSolver(local_neighb=local_neighb)
    #     for name, dataset in competition.items():
    #         dataset = [ task for task in dataset if not len(task['solutions'])]
    #         solver.plot(dataset)
    #     competition.time_taken = time.perf_counter() - competition.time_start
    #     print(competition)

# training   : {'correct': 49, 'total': 416, 'error': 0.8822, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct':  8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:30