#!/usr/bin/env python3
import time

import numpy as np
from humanize import precisedelta
from joblib import delayed, Parallel
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.random_agent_unseeded import random_agent_unseeded
from rng.RandomSeedSearch import RandomSeedSearch

time_start = time.perf_counter()


agents = [ RandomSeedSearch(verbose=2), random_agent_unseeded ]
# agents = [ IrrationalSearchAgent, IrrationalAgent(name='pi') ]
# agents = [ IrrationalAgent(name='pi'), IrrationalAgent(name='pi', offset=1) ]
# agents = [ IrrationalSearchAgent(), IrrationalAgent(),  ]
# agents  = [ RandomSeedSearch(use_stats=False), IrrationalAgent(verbose=False) ]
# agents  = [ RandomSeedSearch(use_stats=False, verbose=2), IrrationalAgent(name='pi', verbose=False) ]

# results = evaluate(
results = Parallel(1)(
    delayed(evaluate)(
        "rps",
        agents,
        configuration={
            "episodeSteps": 1000,
            "actTimeout":   1000,  # For debugging
        },
        num_episodes=1,
        debug=True
    )
    for _ in range(1)
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
