#!/usr/bin/env python3
import gc
import time
from copy import deepcopy

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from core.ConnectX import ConnectX
from core.ConnextXBitboard import ConnectXBitboard
from heuristics.LibertiesHeuristic import LibertiesHeuristic

env = make("connectx", debug=True)
env.configuration.timeout = 24*60*60
observation   = env.state[0].observation
configuration = env.configuration

tests = [
    (list(range(1, 6+1)), ConnectX(observation, configuration, LibertiesHeuristic), AlphaBetaAgent,    ),
    (list(range(1, 9+1)), ConnectXBitboard(observation, configuration, None),       AlphaBetaBitboard, ),
]
for depth_range, game, agent_class in tests:
    game = deepcopy(game)
    print(f'{agent_class.__name__:23s}', end=' | ', flush=True)
    for search_max_depth in depth_range:
        start_time = time.perf_counter()
        agent      = agent_class(game, search_max_depth=search_max_depth, verbose_depth=False)
        agent.iterative_deepening_search()
        time_taken = time.perf_counter() - start_time
        print(f'{search_max_depth}={time_taken:.2f}s', end=' ', flush=True)
        gc.collect()
    print()

### Expensive Functions
### - extensions
### - line_from_position
### - next_coord
### - is_valid_coord
### - cached_property
### - score

### Timings - 2011 MacBook Pro
# AlphaBetaAgent     | 1=1.09s 2=0.13s 3=0.70s 4=2.85s 5=21.01s 6=34.79s - @jit() extensions (slower???)
# AlphaBetaAgent     | 1=0.00s 2=0.02s 3=0.15s 4=0.62s 5=4.50s 6=10.05s  - baseline python
# AlphaBetaAgent     | 1=0.75s 2=0.07s 3=0.36s 4=1.31s 5=7.67s 6=12.91s  - @njit() + next_coord()
# AlphaBetaAgent     | 1=0.21s 2=0.02s 3=0.15s 4=0.64s 5=4.59s 6=8.23s   - @njit() + is_valid_coord()
# AlphaBetaAgent     | 1=1.18s 2=0.02s 3=0.46s 4=0.35s 5=1.13s 6=2.55s   - @njit() + extensions() + liberties()
# AlphaBetaAgent     | 1=0.97s 2=0.02s 3=0.36s 4=0.40s 5=0.98s 6=2.45s   - np.sum() + @njit gameover + utility + extension_score
# AlphaBetaAgent     | 1=1.21s 2=0.03s 3=0.54s 4=0.59s 5=1.43s 6=3.70s   - after bugfixing (without @njit liberties() + extensions()
# AlphaBetaAgent     | 1=4.11s 2=0.06s 3=3.27s 4=1.96s 5=12.18s 6=33.83s - @njit liberties() + extensions() - why so slow???
# AlphaBetaAgent     | 1=0.39s 2=0.03s 3=0.22s 4=1.23s 5=5.93s 6=24.28s  - revert: @njit liberties() + extensions() - why so slow???

### Timings - 2019 Razer Pro
# AlphaBetaAgent     | 1=0.00s 2=0.01s 3=0.10s 4=0.59s 5=2.48s 6=10.42s - pure python
# AlphaBetaAgent     | 1=0.22s 2=0.01s 3=0.12s 4=0.51s 5=2.28s 6=9.25s  - @njit()
# AlphaBetaBitboard  | 1=0.00s 2=0.00s 3=0.01s 4=0.03s 5=0.15s 6=0.67s 7=2.10s 8= 8.88s 9=25.61s  - pure python + original heuristic
# AlphaBetaBitboard  | 1=0.00s 2=0.00s 3=0.01s 4=0.04s 5=0.21s 6=0.91s 7=3.90s 8=15.06s 9=62.96s  - @cached_property   + double_attack_score heuristic
# AlphaBetaBitboard  | 1=0.00s 2=0.00s 3=0.01s 4=0.04s 5=0.16s 6=0.60s 7=2.76s 8= 8.76s 9=34.58s  - @clru_cache(None)  + double_attack_score heuristic
# AlphaBetaBitboard  | 1=0.01s 2=0.00s 3=0.02s 4=0.04s 5=0.17s 6=0.62s 7=3.30s 8=10.20s 9=44.47s  - @clru_cache(2**16) + double_attack_score heuristic
# AlphaBetaBitboard  | 1=0.00s 2=0.00s 3=0.02s 4=0.09s 5=0.43s 6=1.79s 7=7.82s 8=28.63s 9=114.14s - without caching
# AlphaBetaBitboard  | 1=2.34s 2=0.01s 3=0.04s 4=0.18s 5=0.68s 6=2.21s 7=8.38s 8=22.84s           - @njit(nogil=True, parallel=True)
# AlphaBetaBitboard  | 1=0.82s 2=0.00s 3=0.01s 4=0.04s 5=0.19s 6=0.67s 7=3.06s 8=9.48s 9=35.27s   - @njit(nogil=True, parallel=False)
