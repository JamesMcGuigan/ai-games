#!/usr/bin/env python3
import time

from kaggle_environments import make

from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent
from games.connectx.core.ConnectX import ConnectX
from games.connectx.heuristics.LinesHeuristic import LinesHeuristic



env = make("connectx", debug=True)
env.configuration.timeout = 24*60*60
observation   = env.state[0].observation
configuration = env.configuration
game          = ConnectX(observation, configuration, LinesHeuristic)

for search_max_depth in [1,2,3,4,5,6]:
    start_time = time.perf_counter()
    agent      = AlphaBetaAgent(game, search_max_depth=search_max_depth, verbose_depth=False)
    agent.iterative_deepening_search()
    time_taken = time.perf_counter() - start_time
    print(f'{search_max_depth}={time_taken:.2f}s', end=' ', flush=True)


### Expensive Functions
### - extensions
### - line_from_position
### - next_coord
### - is_valid_coord
### - cached_property
### - score

### Timings - 2011 MacBook Pro
# 1=1.09s 2=0.13s 3=0.70s 4=2.85s 5=21.01s 6=34.79s - @jit() extensions (slower???)


# 1=0.00s 2=0.02s 3=0.15s 4=0.62s 5=4.50s 6=10.05s  - baseline python
# 1=0.75s 2=0.07s 3=0.36s 4=1.31s 5=7.67s 6=12.91s  - @njit() + next_coord()
# 1=0.21s 2=0.02s 3=0.15s 4=0.64s 5=4.59s 6=8.23s   - @njit() + is_valid_coord()
# 1=1.18s 2=0.02s 3=0.46s 4=0.35s 5=1.13s 6=2.55s   - @njit() + extensions() + liberties()
# 1=0.97s 2=0.02s 3=0.36s 4=0.40s 5=0.98s 6=2.45s   - np.sum() + @njit gameover + utility + extension_score
# 1=1.21s 2=0.03s 3=0.54s 4=0.59s 5=1.43s 6=3.70s   - after bugfixing (without @njit liberties() + extensions()
# 1=4.11s 2=0.06s 3=3.27s 4=1.96s 5=12.18s 6=33.83s - @njit liberties() + extensions() - why so slow???

