import os
import time
from itertools import product
from typing import List

from fastcache._lrucache import clru_cache
from xgboost import XGBClassifier

from src_james.core.DataModel import Task, Competition
from src_james.ensemble.period import get_period_length0, get_period_length1
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.queries.grid import *
from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_unchanged
from src_james.util.np_cache import np_cache


class XGBSolver(Solver):
    optimise = True
    verbose  = True

    def __init__(self):
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
                self.cache[task.filename] = (xgb,)

    def test(self, task: Task) -> bool:
        """test if the given action correctly solves the task"""
        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)

    def action(self, grid, xgb=None, task=None):
        if task and task.filename not in self.cache: self.fit(task)
        xgb      = xgb or self.cache[task.filename][0]
        features = self.make_features(grid, )
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
    def make_features(cls, grid: np.ndarray):
        nrows, ncols = grid.shape
        features = [
            cls.make_feature(grid, i, j)
            for i,j in product(range(nrows), range(ncols))
        ]
        assert len(set(map(len,features))) == 1
        return np.array(features, dtype=np.int8)

    @classmethod
    @np_cache()
    def make_feature(cls, grid: np.ndarray, i: int, j: int) -> List:
        nrows, ncols = grid.shape
        features = [
            i, j, nrows-i, ncols-j,         # distance from edge
            i+j, i-j, j-i,                  # abs(i-j) can produce worse results
            *grid.shape, nrows*ncols,       # grid shape and pixel size
            grid[i][j],                     # grid[i][j]+1, grid[i][j]-1 = can produce worse results

            *cls.bincount(grid),
            *cls.get_moore_neighbours(grid, i, j),
            *cls.get_tl_tr(grid, i, j),

            query_not_zero(grid,i,j),
            query_max_color(grid,i,j),
            query_min_color(grid,i,j),
            query_max_color_1d(grid,i,j),
            query_min_color_1d(grid,i,j),
            query_count_colors(grid,i,j),
            query_count_colors_row(grid,i,j),
            query_count_colors_col(grid,i,j),
            query_count_squares(grid,i,j),
            query_count_squares_row(grid,i,j),
            query_count_squares_col(grid,i,j),
            max_color_1d(grid),
            min_color_1d(grid),
            get_period_length1(grid),  # has no effect
            get_period_length0(grid),  # has no effect
        ]

        neighbourhoods = [
            grid,
            cls.get_neighbourhood(grid,i,j,1),
            cls.get_neighbourhood(grid,i,j,5),
            grid[i,:], grid[:i+1,:], grid[i:,:],
            grid[:,j], grid[:,:j+1], grid[:,j:],
            grid[i:,j:], grid[:i+1,j:], grid[i:,:j+1], grid[:i+1,:j+1],
        ]
        for neighbourhood in neighbourhoods:
            features += [
                max_color(neighbourhood)           if len(neighbourhood) else 0,
                min_color(neighbourhood)           if len(neighbourhood) else 0,
                count_colors(neighbourhood)        if len(neighbourhood) else 0,
                count_squares(neighbourhood)       if len(neighbourhood) else 0,
            ]
        return features

    @classmethod
    @np_cache()
    def bincount(cls, grid: np.ndarray):
        return np.bincount(grid.flatten(), minlength=11).tolist()  # features requires a fixed length array

    @classmethod
    @np_cache()
    def get_neighbourhood(cls, grid: np.ndarray, i: int, j: int, distance=1):
        try:
            output = np.full((2*distance+1, 2*distance+1), 11)  # 11 = outside of grid pixel
            for xo, xg in enumerate(range(-distance, distance+1)):
                for yo, yg in enumerate(range(-distance, distance+1)):
                    if not 0 <= xo < grid.shape[0]: continue
                    if not 0 <= yo < grid.shape[1]: continue
                    output[xo,yo] = grid[xg,yg]
            return output
        except:
            return np.full((2*distance+1, 2*distance+1), 11)  # 11 = outside of grid pixel

    @classmethod
    @np_cache()
    def get_moore_neighbours(cls, color, cur_row, cur_col):
        nrows, ncols = color.shape
        top    = -1 if cur_row <= 0         else color[cur_row - 1][cur_col]
        bottom = -1 if cur_row >= nrows - 1 else color[cur_row + 1][cur_col]
        left   = -1 if cur_col <= 0         else color[cur_row][cur_col - 1]
        right  = -1 if cur_col >= ncols - 1 else color[cur_row][cur_col + 1]
        return top, bottom, left, right


    @classmethod
    @np_cache()
    def get_tl_tr(cls, color, cur_row, cur_col):
        nrows, ncols = color.shape
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




### Original
# training   : {'correct': 49, 'total': 416, 'error': 0.8822, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct':  8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:30

### Reorder i+j
# training   : {'correct': 50, 'total': 416, 'error': 0.8798, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:27

### Add i+-1, j+-1
# training   : {'correct': 50, 'total': 416, 'error': 0.8798, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:30

### Add i+-j
# training   : {'correct': 54, 'total': 416, 'error': 0.8702, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 20, 'total': 419, 'error': 0.9523, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:28

### Add abs(i-j) - dangerous
# training   : {'correct': 58, 'total': 416, 'error': 0.8606, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 17, 'total': 419, 'error': 0.9594, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 6, 'total': 104, 'error': 0.9423, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:27

### Add color+-1
# training   : {'correct': 50, 'total': 416, 'error': 0.8798, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:31

### max_color(grid),; min_color(grid),; max_color_1d(grid),; min_color_1d(grid),; count_colors(grid),; count_squares(grid),
# training   : {'correct': 88, 'total': 416, 'error': 0.7885, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 57, 'total': 419, 'error': 0.864, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:45

### max_color(grid),; min_color(grid),; max_color_1d(grid),; min_color_1d(grid),; count_colors(grid),; count_squares(grid),
### query_not_zero(grid,i,j),; query_max_color(grid,i,j),; query_min_color(grid,i,j),; query_max_color_1d(grid,i,j),; query_min_color_1d(grid,i,j),; query_count_colors(grid,i,j),  # query_count_colors_row(grid,i,j), query_count_colors_col(grid,i,j), query_count_squares(grid,i,j), # query_count_squares_row(grid,i,j), query_count_squares_col(grid,i,j),
# training   : {'correct': 89, 'total': 416, 'error': 0.7861, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 59, 'total': 419, 'error': 0.8592, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:05

### *grid.shape
# training   : {'correct': 95, 'total': 416, 'error': 0.7716, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 62, 'total': 419, 'error': 0.852, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:18

### *np.bincount(grid.flatten(), minlength=10),; *sorted(np.bincount(grid.flatten(), minlength=10)),
# training   : {'correct': 99, 'total': 416, 'error': 0.762, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 62, 'total': 419, 'error': 0.852, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:29


### *grid.shape, nrows-i, ncols-j,
# training   : {'correct': 109, 'total': 416, 'error': 0.738, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 70, 'total': 419, 'error': 0.8329, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:21

### without bincount
# training   : {'correct': 105, 'total': 416, 'error': 0.7476, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 69, 'total': 419, 'error': 0.8353, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:08

### len(np.bincount(grid.flatten())), *np.bincount(grid.flatten(), minlength=10)
# training   : {'correct': 107, 'total': 416, 'error': 0.7428, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 70, 'total': 419, 'error': 0.8329, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:17



### neighbourhood
# training   : {'correct': 111, 'total': 416, 'error': 0.7332, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 71, 'total': 419, 'error': 0.8305, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:36

# training   : {'correct': 112, 'total': 416, 'error': 0.7308, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 72, 'total': 419, 'error': 0.8282, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 19, 'total': 104, 'error': 0.8173, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:50

### without features += cls.get_neighbourhood(grid,i,j,local_neighbours).flatten().tolist()
# training   : {'correct': 112, 'total': 416, 'error': 0.7308, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 78, 'total': 419, 'error': 0.8138, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 22, 'total': 104, 'error': 0.7885, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:43

### for line in ( grid[i,:], grid[:,j] ):
# training   : {'correct': 127, 'total': 416, 'error': 0.6947, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 87, 'total': 419, 'error': 0.7924, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 22, 'total': 104, 'error': 0.7885, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:07

### for line_neighbourhood in [grid[i,:], grid[:i+1,:], grid[i:,:], grid[:,j], grid[:,:j+1], grid[:,j:], cls.get_neighbourhood(grid,i,j,1), cls.get_neighbourhood(grid,i,j,3), cls.get_neighbourhood(grid,i,j,5),]:
# training   : {'correct': 138, 'total': 416, 'error': 0.6683, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 109, 'total': 419, 'error': 0.7399, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 31, 'total': 104, 'error': 0.7019, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:28

### for neighbourhood in [ grid[i:,j:], grid[:i+1,j:], grid[i:,:j+1], grid[:i+1,:j+1], ]
# training   : {'correct': 148, 'total': 416, 'error': 0.6442, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 116, 'total': 419, 'error': 0.7232, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 33, 'total': 104, 'error': 0.6827, 'time': '00:00:00', 'name': 'test'}
# time       : 00:03:10