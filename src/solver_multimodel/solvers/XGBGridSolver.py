from typing import List

import pydash
from fastcache._lrucache import clru_cache
from itertools import product
from src.ensemble.period import get_period_length0
from src.ensemble.period import get_period_length1
from xgboost import XGBClassifier

from src.datamodel.Competition import Competition
from src.datamodel.Task import Task
from src.functions.queries.grid import *
from src.functions.queries.ratio import is_task_shape_ratio_unchanged
from src.functions.queries.symmetry import is_grid_symmetry
from src.functions.transforms.singlecolor import np_bincount
from src.settings import settings
from src.solver_multimodel.core.Solver import Solver
from src.util.np_cache import np_cache


class XGBGridSolver(Solver):
    optimise = True
    verbose  = True
    xgb_defaults = {
        'tree_method':      'exact',
        'eval_metric':      'error',
        'objective':        'reg:squarederror',
        'n_estimators':     32,
        'max_depth':        100,
        'min_child_weight': 0,
        # 'sampling_method':  'uniform',
        # 'max_delta_step':   1,
        # 'min_child_weight': 0,
    }

    def __init__(self, n_estimators=24, max_depth=10, **kwargs):
        super().__init__()
        self.kwargs = {
            **self.xgb_defaults,
            'n_estimators': n_estimators,
            'max_depth':    max_depth,
            **kwargs,
        }
        if self.kwargs.get('booster') == 'gblinear':
            self.kwargs = pydash.omit(self.kwargs, *['max_depth'])

    def __repr__(self):
        return f'<{self.__class__.__name__}:self.kwargs>'

    def format_args(self, args):
        return self.kwargs

    def detect(self, task):
        if not is_task_shape_ratio_unchanged(task): return False
        # inputs, outputs, not_valid = self.features(task)
        # if not_valid: return False
        return True

    def create_classifier(self, **kwargs):
        kwargs     = { **self.kwargs, **kwargs }
        classifier = XGBClassifier(kwargs)
        return classifier

    def fit(self, task):
        if task.filename not in self.cache:
            # inputs  = task['train'].inputs + task['test'].inputs
            # outputs = task['train'].outputs
            inputs, outputs, not_valid = self.features(task)
            if not_valid:
                self.cache[task.filename] = None
            else:
                # BUGFIX: occasionally throws exceptions in jupyter
                classifier = None
                try:
                    classifier = self.create_classifier()
                    classifier.fit(inputs, outputs, verbose=False)
                    self.cache[task.filename] = (classifier,)
                except Exception as exception:
                    if self.debug:
                        print(f'{self.__class__.__name__}:fit({task}] | Exception: ')
                        print(classifier)
                        print(type(exception), exception)
                    pass

    def test(self, task: Task) -> bool:
        """test if the given solve_grid correctly solves the task"""
        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.solve_grid, *args, task=task)

    def solve_grid(self, grid: np.ndarray, xgb=None, task=None, **kwargs):
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

            # TODO: Reshape all input/outputs to largest size
            if (target_rows != nrows) or (target_cols != ncols):
                return None, None, 1

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

            *np_bincount(grid),
            *grid_unique_colors(grid),
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
            is_grid_symmetry(grid),
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
    def get_neighbourhood(cls, grid: np.ndarray, i: int, j: int, distance=1):
        try:
            output = np.full((2*distance+1, 2*distance+1), 11)  # 11 = outside of grid pixel
            for x_out, x_grid in enumerate(range(-distance, distance+1)):
                for y_out, y_grid in enumerate(range(-distance, distance+1)):
                    if not 0 <= x_out < grid.shape[0]: continue
                    if not 0 <= y_out < grid.shape[1]: continue
                    output[x_out,y_out] = grid[i+x_grid,j+y_grid]
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


class XGBGridSolverDart(XGBGridSolver):
    kwargs_defaults = {
        'booster': 'dart',
        'eval_metric': 'error',
        'grow_policy': 'lossguide',
        'objective': 'reg:squaredlogerror',
        'sampling_method': 'gradient_based',
        'tree_method': 'hist'
    }
    def __init__(self, **kwargs):
        self.kwargs = { **self.kwargs_defaults, **kwargs }
        super().__init__(**self.kwargs)

class XGBGridSolverGBtree(XGBGridSolver):
    kwargs_defaults = {
        'booster': 'gbtree',
        'eval_metric': 'ndcg',
        'grow_policy': 'depthwise',
        'objective': 'reg:squarederror',
        'sampling_method': 'uniform',
        'tree_method': 'exact'
    }
    def __init__(self, booster='gbtree', **kwargs):
        self.kwargs = { "booster": booster, **self.kwargs_defaults, **kwargs }
        super().__init__(**self.kwargs)

class XGBGridSolverGBlinear(XGBGridSolver):
    def __init__(self, booster='gblinear', **kwargs):
        self.kwargs = { "booster": booster, "max_depth": None, **kwargs }
        super().__init__(**self.kwargs)


if __name__ == '__main__' and not settings['production']:
    solver = XGBGridSolver()
    solver.verbose = True
    competition = Competition()
    competition.map(solver.solve_dataset)
    print(competition)




### Original
# training   : {'correct': 18, 'guesses': 49, 'total': 416, 'error': 0.9567, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 3, 'guesses': 19, 'total': 419, 'error': 0.9928, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 8, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}

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

# XGBGridSolver(n_estimators=10)
# training   : {'correct': 22, 'guesses': 148, 'total': 416, 'error': 0.9471, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 9, 'guesses': 116, 'total': 419, 'error': 0.9785, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 33, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:53

# XGBGridSolver(n_estimators=32)
# training   : {'correct': 25, 'guesses': 255, 'total': 416, 'error': 0.9399, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 10, 'guesses': 257, 'total': 419, 'error': 0.9761, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 64, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:39

### XGBSolverDart(); XGBSolverGBtree(); XGBSolverGBlinear()
# training   : {'correct': 42, 'guesses': 254, 'total': 416, 'error': 0.899, 'time': '00:03:31', 'name': 'training'}
# evaluation : {'correct': 14, 'guesses': 242, 'total': 419, 'error': 0.9666, 'time': '00:07:17', 'name': 'evaluation'}
# test       : {'correct': 3.5, 'guesses': 61, 'total': 104, 'error': 1.0, 'time': '00:01:35', 'name': 'test'}
# time       : 00:12:23

### max_depth=10
# training   : {'correct': 43, 'guesses': 266, 'total': 416, 'error': 0.8966, 'time': '00:04:49', 'name': 'training'}
# evaluation : {'correct': 14, 'guesses': 266, 'total': 419, 'error': 0.9666, 'time': '00:08:23', 'name': 'evaluation'}
# test       : {'correct': 3.4, 'guesses': 65, 'total': 104, 'error': 1.0, 'time': '00:01:40', 'name': 'test'}
# time       : 00:14:53
