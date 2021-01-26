# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-seed-search

import random
import re
import time
from typing import List

import numpy as np
import tensorflow as tf
from mpmath import mp

history     = []
min_seed    = 0
best_method = 'random'
solutions   = []


# Irrational numbers are pure random sequences that are immune to random seed search
mp.dps     = random.randint(1234,9876)
irrational = mp.pi()  # mp.e() + mp.pi() + mp.sqrt(2) + mp.euler()
irrational = re.sub('[^1-9]', '', str(irrational))
# irrational = irrational[::-1]
# print(f'len(irrational) = {len(irrational)}\n{irrational[:1000]}\n')
def random_agent(observation, configuration, seed=None):
    return (int(irrational[observation.step]) + 1) % configuration.signs  # Anti-PI bot


# https://github.com/JamesMcGuigan/kaggle-digit-recognizer/blob/master/src/random/random_seed_search.py
def get_random(length, seed, method='random') -> List[int]:
    if method == 'random':
        random.seed(seed)
        return [ random.randint(0,2) for n in range(length) ]
    if method == 'np':
        np.random.seed(seed)
        return np.random.randint(0,2, length).tolist()
    if method == 'tf':
        tf.random.set_seed(seed)
        return tf.random.uniform((length,), minval=0, maxval=3, dtype=tf.dtypes.int32).numpy().tolist()


# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def random_seed_search_agent(observation, configuration, warmup=10, seeds_per_turn=200_000):
    # print(observation)
    global min_seed, best_method, solutions
    time_start      = time.perf_counter()
    # time_end      = time_start + configuration.actTimeout - safety_time
    opponent_action = observation.lastOpponentAction if observation.step > 0 else None
    if opponent_action is not None:
        history.append(opponent_action)

    # Play the first few rounds as pure random to see what the opponent does
    if observation.step <= warmup:
        return random_agent(observation, configuration, seed=None)

    action = random_agent(observation, configuration, seed=None)
    try:
        # Search through the list of previously found solutions and see if any still match
        for seed, method in solutions:
            guess      = get_random(length=len(history)+1, seed=seed, method=method)
            prediction = guess[-1]

            # BUG: when seed gets reused, opponent seems to skip a step
            if guess[:-1] == history:
                action = (prediction + 1) % configuration.signs
                print(f'Reused Seed: {method} {seed} | action = {action} | {solutions}')
                return int(action)
                # break
        else:
            # Continue search for seeds until timeout
            methods    = [ 'random', 'np', 'tf' ]
            loop_count = int( seeds_per_turn / len(methods) / len(history) )
            for seed in range(min_seed, min_seed + loop_count):
                for method in methods:
                    guess      = get_random(length=len(history)+1, seed=seed, method=method)
                    prediction = int(guess[-1])
                    if guess[:-1] == history:
                        solutions += [ (seed, method) ]
                        action     = (prediction + 1) % configuration.signs
                        print(f'Found  Seed: {method} {seed} | action = {action} | {solutions} | {history}')
                        return int(action)
                        # break
                    else:
                        min_seed += 1
            else:
                action = random_agent(observation, configuration, seed=None)
    except Exception as exception:
        print(exception)

    time_taken = time.perf_counter() - time_start
    print(f'time = {time_taken:.2f}s | step = {observation.step:4d} | action = {action} | solutions = {solutions}')
    return int(action) % configuration.signs
