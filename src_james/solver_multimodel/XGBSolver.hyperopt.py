# DOCS: https://stackoverflow.com/questions/43927725/python-hyperparameter-optimization-for-xgbclassifier-using-randomizedsearchcv
# DOCS: https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb
import os
import random
import time

import numpy as np
from hyperopt import fmin
from hyperopt import hp
from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt.pyll import scope

from src_james.core.DataModel import Competition
from src_james.solver_multimodel.XGBSolver import XGBSolver
from src_james.util.timeout import timeout


def resolve_hparms(best_param, param_space):
    best_param_resolved = {}
    for key, value in best_param.items():
        resolved = value
        if isinstance(value, (int,np.int64)) and isinstance(param_space[key], type(hp.choice('',[]))):
            try: resolved = param_space[key].inputs()[value].eval()
            except: pass

        if isinstance(resolved, (int,np.int64)):
            resolved = int(resolved)
        if isinstance(resolved, (float,np.float64)):
            resolved = round(float(resolved), 4)
        best_param_resolved[key] = resolved
    return best_param_resolved


def hyperopt(param_space, num_eval, verbose=True, timeout_seconds=120):
    start = time.time()

    def test_function(params):
        competition = Competition()
        dataset     = competition['evaluation']
        try:
            with timeout(timeout_seconds):
                solver = XGBSolver(**params)
                solver.verbose = False
                # competition['test'].apply(solver.solve_all)  # Quick for debugging
                # competition.map(solver.solve_all)
                dataset.apply(solver.solve_all)
        except Exception as exception:
            # if verbose:
            #     print('Exception:')
            #     print(params)
            #     print(type(exception), exception)
            pass
        return dataset

    def objective_function(params):
        time_start = time.perf_counter()
        dataset    = test_function(params)
        time_taken = time.perf_counter() - time_start
        score = sum([
            # competition.score()['training']['correct'],
            # competition.score()['evaluation']['correct'],
            # competition.score()['test']['correct'] * 3,
            dataset.score()['correct'],
            -int(time_taken)/1000,
        ])
        params = { 'score': score, "time": int(time_taken), **params }
        params = { key: round(value,3) if isinstance(value,float) else value for key,value in params.items() }
        if verbose:
            print(params)
            # print(competition)
        status = STATUS_OK if score > 0 and time_taken < timeout_seconds else STATUS_FAIL
        return {'loss': -score, 'status': status}

    trials = Trials()
    best_param = fmin(
        objective_function,
        param_space,
        algo=tpe.suggest,
        max_evals=num_eval,
        trials=trials,
        rstate= np.random.RandomState(random.randint(0,10000)),
        max_queue_len=8,
    )

    loss = [x['result']['loss'] for x in trials.trials]

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
    print(test_function(best_param))

    competition = Competition()
    dataset     = competition['evaluation']
    solver = XGBSolver(**best_param)
    solver.verbose = False
    competition.map(solver.solve_all)
    print(competition)
    return trials

if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    # DOCS: https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    param_hyperopt = {
        # 'n_estimators':      hp.choice('n_estimators', [8]),  # 24+ is optimal, but 8 is quicker once cached
        'n_estimators':      scope.int(hp.quniform('n_estimators',      16, 40, 1)),
        'num_parallel_tree': scope.int(hp.quniform('num_parallel_tree', 1,  40, 1)),
        'booster':           hp.choice('booster', [
            'gbtree',
            'dart',
            # 'gblinear', # WARNING: Parameters: { colsample_bytree, max_delta_step, max_depth, min_child_weight, num_parallel_tree, subsample } might not be used.
        ]),
        'min_split_loss':      hp.loguniform('learning_rate',    np.log(1e-10), np.log(1e10)),
        # 'scale_pos_weight':    hp.loguniform('scale_pos_weight', np.log(1e-10), np.log(1e10)),
        'max_depth':           scope.int(hp.quniform('max_depth', 5, 25, 1)),
        'min_child_weight':    scope.int(hp.quniform('min_child_weight', 0, 4,  1)),
        'max_delta_step':      scope.int(hp.quniform('max_delta_step',   0, 10, 1)),
        'tree_method':         hp.choice('tree_method', ['exact', 'approx', 'hist']),
        # 'updater':             hp.choice('objective', ['grow_colmaker', 'grow_histmaker', 'grow_local_histmaker', 'grow_skmaker', 'grow_quantile_histmaker', 'grow_gpu_hist', 'refresh','prune']),
        # 'learning_rate':     hp.loguniform('learning_rate', np.log(0.01), np.log(1)),

        # 'colsample_bytree':  hp.uniform('colsample_by_tree', 0.6, 1.0),
        # 'reg_lambda':        hp.uniform('reg_lambda', 0.0, 1.0),
        # 'subsample':         hp.uniform('subsample', 0.3, 1.0),
        # 'max_delta_step':    scope.int(hp.quniform('max_delta_step', 0, 10, 1)),
        # 'num_parallel_tree': scope.int(hp.quniform('num_parallel_tree', 1, 5, 1)),
        # 'base_score':        hp.uniform('base_score', 0.25, 0.75),
        'objective':         hp.choice('objective', [
            'reg:squarederror',
            'reg:squaredlogerror',
            'reg:logistic',
            # 'reg:pseudohubererror',  # Unknown objective function: `reg:pseudohubererror`
            'reg:gamma',
            'reg:tweedie',
            'count:poisson',
            'survival:cox',
            'survival:aft',
            # 'multi:softmax',   # wrong datastructure
            # 'multi:softprob',  # wrong datastructure
            'rank:pairwise',
            # 'rank:ndcg',       # worse than rank:pairwise'
            # 'rank:map',        # worse than rank:pairwise'
        ]),
        "eval_metric": hp.choice('eval_metric', [
            'rmse', 'rmsle', 'mae', 'mphe', 'logloss', 'error', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg', 'map',
            'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik', 'gamma-deviance', 'tweedie-nlogli', 'aft-nloglik'
        ])
    }
    hyperopt(param_hyperopt, 25, verbose=True, timeout_seconds=120)


# {'score': 71.0, 'base_score': 0.5580316005760565, 'booster': 'gbtree', 'colsample_bytree': 0.8711693954468235, 'learning_rate': 0.26376465700523233, 'max_delta_step': 5, 'max_depth': 11, 'min_child_weight': 1, 'n_estimators': 8, 'num_parallel_tree': 4, 'objective': 'reg:tweedie', 'reg_lambda': 0.29549016341486206, 'subsample': 0.9863675458668577}
