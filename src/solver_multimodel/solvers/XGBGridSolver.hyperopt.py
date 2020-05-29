# DOCS: https://stackoverflow.com/questions/43927725/python-hyperparameter-optimization-for-xgbclassifier-using-randomizedsearchcv
# DOCS: https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb
import os
import random
from copy import deepcopy

import numpy as np
import time
from hyperopt import fmin
from hyperopt import hp
from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt.pyll import scope

from src.datamodel.Competition import Competition
from src.solver_multimodel.solvers.XGBGridSolver import XGBGridSolver
from src.util.timeout import timeout


def resolve_hparms(best_param, param_space):
    best_param_resolved = {}
    for key, value in best_param.items():
        if isinstance(value, (int,np.int32,np.int64)):
            value = int(value)
        if isinstance(value, (float,np.float32,np.float64)):
            value = round(float(value), 4)
        if isinstance(param_space[key], type(hp.choice('',[]))):
            try: value = param_space[key].inputs()[value].eval()
            except:
                try:    value = param_space[key].pos_args[value+1].obj
                except: print('resolve_hparms()', type(value), value, param_space[key])

        best_param_resolved[key] = value
    return best_param_resolved


# Compiles, but doesn't seem to have any impact
def encode_points_to_evaluate(points_to_evaluate, param_space):
    if points_to_evaluate is None: return None

    points_to_evaluate = deepcopy(points_to_evaluate)
    for params in points_to_evaluate:
        for key, value in params.items():
            if key not in param_space.keys(): del params[key]; continue
            if isinstance(param_space[key], type(hp.choice('',[]))):
                for index, pos_arg in enumerate(param_space[key].pos_args[1:]):  # First pos_arg is namespace
                    if value == pos_arg.obj:
                        params[key] = index
                        break
    return points_to_evaluate



def hyperopt(param_space, num_eval, points_to_evaluate=None, verbose=True, timeout_seconds=120, debug=False):
    start = time.time()

    def test_function(params={}):
        competition = Competition()
        try:
            if not debug:
                with timeout(timeout_seconds):
                    solver = XGBGridSolver(verbosity=0, **params)
                    solver.verbose = False
                    # competition['test'].apply(solver.solve_dataset)  # Quick for debugging
                    # competition.map(solver.solve_dataset)
                    competition['training'].apply(solver.solve_dataset)
                    competition['evaluation'].apply(solver.solve_dataset)
        except Exception as exception:
            # if verbose:
            #     print('Exception:')
            #     print(params)
            #     print(type(exception), exception)
            pass
        return competition

    def objective_function(params):
        time_start  = time.perf_counter()
        competition = test_function(params)
        time_taken  = time.perf_counter() - time_start
        score = sum([
            competition.score()['training']['correct']/100,
            competition.score()['evaluation']['correct'],
            # competition.score()['test']['correct'] * 3,
            # dataset.score()['correct'],
            # -int(time_taken)/1000,
        ])
        params = { 'score': score, "time": int(time_taken), **params }
        params = { key: round(value,3) if isinstance(value,float) else value for key,value in params.items() }
        if verbose:
            print(params)
            # print(competition)
        status = STATUS_OK if debug or score > 0 else STATUS_FAIL
        return {'loss': -score, 'status': status}

    points = encode_points_to_evaluate(points_to_evaluate, param_space)
    trials = Trials()
    best_param = fmin(
        objective_function,
        param_space,
        algo=tpe.suggest,
        max_evals=num_eval,
        trials=None, # trials,
        rstate= np.random.RandomState(random.randint(0,10000)),
        max_queue_len=8,
        points_to_evaluate=points,
    )

    loss = [x['result']['loss'] for x in trials.trials] if len(trials.trials) else [0]

    best_param = resolve_hparms(best_param, param_space)

    print("-"*50)
    print("##### Results")
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    print("Score best parameters: ", min(loss)*-1)
    print("-"*20)
    print("Best parameters: ")
    print()
    for key, value in best_param.items():
        print(f'{key} =', value, type(value).__name__)
    # print("Test Score: ", score)
    print("-"*20)

    competition = Competition()
    dataset     = competition['evaluation']
    solver = XGBGridSolver(**best_param)
    solver.verbose = False
    competition.map(solver.solve_dataset)
    print(competition)
    return trials

if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    # DOCS: https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    param_hyperopt = {
        # 'n_estimators':        hp.choice('n_estimators', [8]),  # 24+ is optimal, but 8 is quicker once cached
        'n_estimators':        scope.int(hp.quniform('n_estimators',      16, 40, 1)),
        'num_parallel_tree':   scope.int(hp.quniform('num_parallel_tree', 1,  40, 1)),
        'max_depth':           scope.int(hp.quniform('max_depth', 5, 25, 1)),
        'min_split_loss':      hp.loguniform('min_split_loss',   np.log(1e-3), np.log(1e3)),
        'scale_pos_weight':    hp.loguniform('scale_pos_weight', np.log(1e-3), np.log(1e3)),
        'min_child_weight':    scope.int(hp.quniform('min_child_weight', 0, 4,  1)),
        'max_delta_step':      scope.int(hp.quniform('max_delta_step',   0, 10, 1)),
        'learning_rate':       hp.loguniform('learning_rate', np.log(0.01), np.log(100)),
        # 'colsample_bytree':  hp.uniform('colsample_by_tree', 0.6, 1.0),
        # 'reg_lambda':          hp.uniform('reg_lambda', 0.0, 1.0),
        # 'subsample':         hp.uniform('subsample', 0.3, 1.0),
        # 'max_delta_step':    scope.int(hp.quniform('max_delta_step', 0, 10, 1)),
        # 'num_parallel_tree': scope.int(hp.quniform('num_parallel_tree', 1, 5, 1)),
        # 'base_score':        hp.uniform('base_score', 0.25, 0.75),
        # 'updater':           hp.choice('objective', ['grow_colmaker', 'grow_histmaker', 'grow_local_histmaker', 'grow_skmaker', 'grow_quantile_histmaker', 'grow_gpu_hist', 'refresh','prune']),
        'sampling_method':     hp.choice('sampling_method', [
            'uniform',
            'gradient_based'
        ]),
        'grow_policy':         hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'tree_method':         hp.choice('tree_method', [
                'exact',
            # 'approx',
            # 'hist',
            # 'gpu_hist'  # score 0
        ]),
        'booster':             hp.choice('booster', [
            'gbtree',
            # 'dart',
            # 'gblinear', # WARNING: Parameters: { colsample_bytree, max_delta_step, max_depth, min_child_weight, num_parallel_tree, subsample } might not be used.
        ]),
        'objective':         hp.choice('objective', [
            'reg:squarederror',    # score 9
            # 'reg:squaredlogerror', # score 9
            # 'reg:logistic',
            # 'reg:pseudohubererror',  # Unknown objective function: `reg:pseudohubererror`
            # 'reg:gamma',
            # 'reg:tweedie',       # score 8+
            # 'count:poisson',     # score 9
            # 'survival:cox',
            # 'survival:aft',    # score 8+
            # 'multi:softmax',   # wrong datastructure
            # 'multi:softprob',  # wrong datastructure
            # 'rank:pairwise',   # score 0
            # 'rank:ndcg',       # worse than rank:pairwise'
            # 'rank:map',        # worse than rank:pairwise'
        ]),
        "eval_metric": hp.choice('eval_metric', [
            # 'rmse',          # score 0
            # 'rmsle',         # score 0
            # 'mae',           # score 0
            # 'mphe',          # score 0
            # 'logloss',         # score 9
            'error',           # score 9
            # 'merror',          # score 8+
            # 'mlogloss',      # score 0
            # 'auc',           # score 0
            # 'aucpr',           # score 9
            # 'ndcg',            # score 9
            # 'map',
            # 'poisson-nloglik', # score 9
            # 'gamma-nloglik',   # score 9
            # 'cox-nloglik',
            # 'gamma-deviance',
            # 'tweedie-nlogli',
            # 'aft-nloglik'     # score 0
        ])
    }
    ### Best keyword results - score = 9
    # points_to_evaluate = [
    #     { 'booster': 'dart',   'eval_metric': 'error',          'grow_policy': 'lossguide', 'objective': 'reg:squaredlogerror', 'sampling_method': 'gradient_based', 'tree_method': 'hist'},
    #     { 'booster': 'gbtree', 'eval_metric': 'ndcg',           'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'exact'},
    #     { 'booster': 'gbtree', 'eval_metric': 'aucpr',          'grow_policy': 'lossguide', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'exact'},
    #     { 'booster': 'gbtree', 'eval_metric': 'ndcg',           'grow_policy': 'lossguide', 'objective': 'count:poisson',       'sampling_method': 'uniform',        'tree_method': 'exact'},
    #     { 'booster': 'dart',   'eval_metric': 'gamma-nloglik',  'grow_policy': 'lossguide', 'objective': 'count:poisson',       'sampling_method': 'gradient_based', 'tree_method': 'approx'},
    #     { 'booster': 'gbtree', 'eval_metric': 'aucpr',          'grow_policy': 'lossguide', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'approx'},
    #     { 'booster': 'dart',   'eval_metric': 'logloss',        'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'gradient_based', 'tree_method': 'approx'},
    #     { 'booster': 'dart',   'eval_metric': 'gamma-deviance', 'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'gradient_based', 'tree_method': 'approx'},
    # ]
    points_to_evaluate = None
    hyperopt(param_hyperopt, 50, points_to_evaluate=points_to_evaluate, verbose=True, timeout_seconds=180)

### The problem with running hyperopt on XGBoost for this dataset is that the results are highly variable - 9s score 4-7 on rerun
### Best keyword results - score = 9
# {'score': 9, 'time': 85,  'booster': 'dart',   'eval_metric': 'error',          'grow_policy': 'lossguide', 'objective': 'reg:squaredlogerror', 'sampling_method': 'gradient_based', 'tree_method': 'hist'}
# {'score': 9, 'time': 94,  'booster': 'gbtree', 'eval_metric': 'ndcg',           'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'exact'}
# {'score': 9, 'time': 135, 'booster': 'gbtree', 'eval_metric': 'aucpr',          'grow_policy': 'lossguide', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'exact'}
# {'score': 9, 'time': 173, 'booster': 'gbtree', 'eval_metric': 'ndcg',           'grow_policy': 'lossguide', 'objective': 'count:poisson',       'sampling_method': 'uniform',        'tree_method': 'exact'}
# {'score': 9, 'time': 180, 'booster': 'dart',   'eval_metric': 'gamma-nloglik',  'grow_policy': 'lossguide', 'objective': 'count:poisson',       'sampling_method': 'gradient_based', 'tree_method': 'approx'}
# {'score': 9, 'time': 200, 'booster': 'gbtree', 'eval_metric': 'aucpr',          'grow_policy': 'lossguide', 'objective': 'reg:squarederror',    'sampling_method': 'uniform',        'tree_method': 'approx'}
# {'score': 9, 'time': 206, 'booster': 'dart',   'eval_metric': 'logloss',        'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'gradient_based', 'tree_method': 'approx'}
# {'score': 9, 'time': 231, 'booster': 'dart',   'eval_metric': 'gamma-deviance', 'grow_policy': 'depthwise', 'objective': 'reg:squarederror',    'sampling_method': 'gradient_based', 'tree_method': 'approx'}


### Figure out broken keys | -bad +good
# for key in 0 '[4-7]' '[8-9]' 9; do cat src/solver_multimodel/logs/*keyword* | grep "score': $key" | perl -p -e "s/': '/:/g; s/['{}\d,]//g;"  | perl -p -e "s/\s/\n/g; s/\w+:\W//g;" | sed -r '/^\s*$/d' | sort | uniq -c | sort -k2 > src/solver_multimodel/logs/score/keywords.$key.txt; done;
# diff <(awk2 src/solver_multimodel/logs/score/keywords.0.txt) <(awk2 src/solver_multimodel/logs/score/keywords.\[8-9\].txt) -u | grep -v '^ '
# -booster:gblinear
# -eval_metric:aft-nloglik
# -eval_metric:auc
# -eval_metric:mae
# -eval_metric:mlogloss
# -eval_metric:mphe
# -eval_metric:rmse
# -eval_metric:rmsle
# -eval_metric:tweedie-nlogli
# +eval_metric:merror
# +eval_metric:ndcg
# +eval_metric:poisson-nloglik
# -objective:multi:softmax
# -objective:multi:softprob
# -objective:rank:map
# +objective:count:poisson
# -objective:rank:pairwise
# -objective:survival:aft
# +objective:reg:tweedie
# -tree_method:gpu_hist

### Parameters capable of scoring 9: cat src/solver_multimodel/logs/score/keywords.9.txt
# 4 booster:dart
# 4 booster:gbtree
# 2 eval_metric:aucpr
# 1 eval_metric:error
# 1 eval_metric:gamma-deviance
# 1 eval_metric:gamma-nloglik
# 1 eval_metric:logloss
# 2 eval_metric:ndcg
# 3 grow_policy:depthwise
# 5 grow_policy:lossguide
# 2 objective:count:poisson
# 5 objective:reg:squarederror
# 1 objective:reg:squaredlogerror
# 4 sampling_method:gradient_based
# 4 sampling_method:uniform
# 4 tree_method:approx
# 3 tree_method:exact
# 1 tree_method:hist
