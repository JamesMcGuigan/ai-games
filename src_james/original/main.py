import time

from src_james.core.DataModel import Competition
from src_james.original.solvers import solvers



time_start   = time.perf_counter()
competition  = Competition()
scores       = { name: 0 for name in competition.values() }
for name, dataset in competition.items():
    for solver in solvers:
        solver.cache = {}
        scores[name] = solver.solve_all(dataset)

competition.score()
