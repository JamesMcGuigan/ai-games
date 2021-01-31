import time

import numpy as np
from humanize import precisedelta
from joblib import delayed, Parallel
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.IrrationalSearchAgent import IrrationalSearchAgent
from rng.random_agent_unseeded import random_agent_unseeded
from rng.RandomSeedSearch import random_seed_search_agent, RandomSeedSearch

time_start = time.perf_counter()

agents  = [ RandomSeedSearch(), random_agent_unseeded ]
agents  = [ IrrationalSearchAgent(), IrrationalAgent(name='pi') ]
results = Parallel(-1)(
    delayed(evaluate)(
        "rps",
        agents,
        configuration={
            "episodeSteps": 1000,
            # "actTimeout":   1,
        },
        num_episodes=1,
        debug=False
    )
    for _ in range(80)
)
results = np.array(results).reshape((-1,2))

time_taken = time.perf_counter() - time_start
print([ getattr(agent, '__name__', agent.__class__.__name__) for agent in agents ])
print('winrate', [ np.sum(results[:,0] > results[:,1]+20), np.sum(results[:,0]+20 < results[:,1]) ], '/', len(results))
print('scores ', np.sum(results, axis=0).round(2))
print('std    ', np.std(results, axis=0).round(2))
print('time:  ', precisedelta(time_taken))
