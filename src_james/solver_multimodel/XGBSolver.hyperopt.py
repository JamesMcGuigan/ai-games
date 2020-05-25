# DOCS: https://stackoverflow.com/questions/43927725/python-hyperparameter-optimization-for-xgbclassifier-using-randomizedsearchcv
# DOCS: https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb
import os
import time

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from src_james.core.DataModel import Competition
from src_james.solver_multimodel.XGBSolver import XGBSolver


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


def hyperopt(param_space, num_eval, verbose=True):
    start = time.time()

    def test_function(params):
        competition = Competition()
        try:
            solver = XGBSolver(**params)
            solver.verbose = False
            # competition['test'].apply(solver.solve_all)  # Quick for debugging
            competition.map(solver.solve_all)
        except Exception as exception:
            print('Exception:')
            print(params)
            print(type(exception), exception)
        return competition

    def objective_function(params):
        competition = test_function(params)
        score = sum([
            competition.score()['training']['correct'],
            competition.score()['evaluation']['correct'] * 4,
            competition.score()['test']['correct'] * 4,
        ])
        params = { 'score': score, **params }
        if verbose:
            print(params)
            print(competition)
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function,
                      param_space,
                      algo=tpe.suggest,
                      max_evals=num_eval,
                      trials=trials,
                      rstate= np.random.RandomState(1))
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
    return trials

if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    # DOCS: https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    param_hyperopt = {
        'n_estimators':      hp.choice('n_estimators', [8]),  # 24+ is optimal, but 8 is quicker once cached
        # 'n_estimators':      scope.int(hp.quniform('n_estimators', 16, 40, 1)),
        'booster':           hp.choice('booster', [
            'gbtree',
            'dart',
            # 'gblinear', # WARNING: Parameters: { colsample_bytree, max_delta_step, max_depth, min_child_weight, num_parallel_tree, subsample } might not be used.
        ]),
        'learning_rate':     hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'max_depth':         scope.int(hp.quniform('max_depth', 5, 15, 1)),
        'colsample_bytree':  hp.uniform('colsample_by_tree', 0.6, 1.0),
        'reg_lambda':        hp.uniform('reg_lambda', 0.0, 1.0),
        'subsample':         hp.uniform('subsample', 0.3, 1.0),
        'min_child_weight':  scope.int(hp.quniform('min_child_weight', 1, 4, 1)),
        'max_delta_step':    scope.int(hp.quniform('max_delta_step', 0, 10, 1)),
        'num_parallel_tree': scope.int(hp.quniform('num_parallel_tree', 1, 5, 1)),
        'base_score':        hp.uniform('base_score', 0.25, 0.75),
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
    }
    hyperopt(param_hyperopt, 50)


{'score': 71.0, 'base_score': 0.5580316005760565, 'booster': 'gbtree', 'colsample_bytree': 0.8711693954468235, 'learning_rate': 0.26376465700523233, 'max_delta_step': 5, 'max_depth': 11, 'min_child_weight': 1, 'n_estimators': 8, 'num_parallel_tree': 4, 'objective': 'reg:tweedie', 'reg_lambda': 0.29549016341486206, 'subsample': 0.9863675458668577}
