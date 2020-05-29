import gc
import time
from operator import itemgetter

from src.core.DataModel import Competition
from src.settings import settings
from src.solver_multimodel.solvers import solvers

if __name__ == '__main__':
    print('\n','-'*20,'\n')
    print('Abstraction and Reasoning Challenge')
    print('Team: Mathematicians + Experts')
    print('https://www.kaggle.com/c/abstraction-and-reasoning-challenge')
    print('\n','-'*20,'\n')
    for solver in solvers: print(solver.__class__.__name__)
    print('\n','-'*20,'\n')

    plot_results = not settings['production']
    time_start   = time.perf_counter()
    competition  = Competition()
    scores       = { name: { solver.__class__.__name__: 0 for solver in solvers } for name in competition.keys() }
    for dataset_name, dataset in competition.items():
        time_start_dataset = time.perf_counter()
        for solver in solvers:
            print('#######', dataset_name, solver.__class__.__name__)
            if plot_results:
                scores[dataset_name][solver.__class__.__name__] += solver.plot(dataset)
            else:
                scores[dataset_name][solver.__class__.__name__] += solver.solve_dataset(dataset)
            # Running on Kaggle uses up nearly all 16GB of RAM
            solver.cache = {}
            gc.collect()

        dataset.time_taken = time.perf_counter() - time_start_dataset
    competition.time_taken = time.perf_counter() - time_start

    competition['test'].write_submission('submission5.csv')
    competition['test'].write_submission()

    print('-'*20)
    print('Solver Scores:')
    for dataset_name in scores.keys():
        print(f'\n# {dataset_name}')
        for solver_name, score in sorted(scores[dataset_name].items(), key=itemgetter(1), reverse=True):
            if score: print(score, solver_name)
    print('-'*20)
    print('Dataset Scores:')
    print(competition)

