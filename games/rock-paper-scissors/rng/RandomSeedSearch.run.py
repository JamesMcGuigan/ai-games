#!/usr/bin/env python3
import time

import numpy as np
from humanize import precisedelta
from joblib import delayed, Parallel
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.RandomSeedSearch import RandomSeedSearch

time_start = time.perf_counter()


# agents = [ RandomSeedSearch, random_agent_unseeded ]
# agents = [ IrrationalAgent(name='pi'), IrrationalSearchAgent ]
# agents = [ IrrationalAgent(name='pi'), IrrationalAgent(name='pi', offset=1) ]
# agents = [ IrrationalAgent(), IrrationalSearchAgent() ]
agents  = [ IrrationalAgent(verbose=False), RandomSeedSearch(use_stats=False) ]

# results = evaluate(
results = Parallel(-1)(
    delayed(evaluate)(
        "rps",
        agents,
        configuration={
            "episodeSteps": 50,
            "actTimeout":   1000,  # For debugging
        },
        num_episodes=100,
        debug=True
    )
    for _ in range(80)
)
results = np.array(results).reshape((-1,2))
results[ results == None ] = 0

time_taken = time.perf_counter() - time_start
print([ getattr(agent, '__name__', agent.__class__.__name__) for agent in agents ])
print('winrate', [ np.sum(results[:,0]-20 > results[:,1]),
                   np.sum(results[:,0]+20 < results[:,1])
                 ], '/', len(results))
print('totals ', np.sum(results, axis=0))
print('std    ', np.std(results, axis=0).round(2))
print('time:  ', precisedelta(time_taken))
