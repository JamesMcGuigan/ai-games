import os
import time

from src_james.core.DataModel import Competition
from src_james.solver_multimodel.solvers import solvers

time_start   = time.perf_counter()
competition  = Competition()
scores       = { name: 0 for name in competition.values() }
for name, dataset in competition.items():
    time_start_dataset = time.perf_counter()
    for solver in solvers:
        solver.cache = {}
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
            scores[name] = solver.solve_all(dataset)
        else:
            scores[name] = solver.plot(dataset)

    dataset.time_taken = time.perf_counter() - time_start_dataset
competition.time_taken = time.perf_counter() - time_start

competition['test'].write_submission()

print('-'*20)
print('Score:')
for key, value in competition.score().items(): print(f'{key.rjust(11)}: {value}')
