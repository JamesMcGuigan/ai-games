# %%writefile main.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-seed-search/
# Source: https://github.com/JamesMcGuigan/ai-games/blob/master/games/rock-paper-scissors/rng/random_seed_search.py

import os
import random
import re
import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from mpmath import mp

### Unsure if setting the seed here affects the opponents RNG in the submission environment
### I think it doesn't and that kaggle submissions environment runs both agents in seperate processes
opponent_seed = 42
random.seed(opponent_seed)
np.random.seed(opponent_seed)
tf.random.set_seed(opponent_seed)



# Irrational numbers are pure random sequences that are immune to random seed search
# DOCS: https://mpmath.org/doc/current/functions/constants.html
mp.dps      = 1234  # slightly more than 1000 as 10%+ of chars will be dropped
irrationals = {
    f'{name}+{n}': list(map(
        lambda c: (int(c)+n) % 3,
        re.sub('[^1-9]', '', str(irrational))
    ))[:1000]
    for n in [0,1,2]
    for name, irrational in {
        'pi':        mp.pi(),
        'phi':       mp.phi(),
        'e':         mp.e(),
        'sqrt2':     mp.sqrt(2),
        'sqrt5':     mp.sqrt(5),
        'euler':     mp.euler(),
        'catalan':   mp.catalan(),
        'apery':     mp.apery(),
        # 'khinchin':  mp.khinchin(),  # slow
        # 'glaisher':  mp.glaisher(),  # slow
        # 'mertens':   mp.mertens(),   # slow
        # 'twinprime': mp.twinprime(), # slow
    }.items()
}
def irrational_agent(observation, configuration, method='pi+0'):
    if f'{method}+1' in irrationals: method = f'{method}+0'
    if method    not in irrationals: method = f'pi+0'

    action = int(irrationals[method][observation.step]) % configuration.signs
    # print(f'Irrational Agent {method} = {action}')
    return action



# 100Mb filesize limit, but tar.gz gives 4.35x compression (6 bits in every int8 are 0)
# Only cache the first ~50 numbers in each sequence,
# Enough for a statistical advantage via hash collisions over Mersenne Twister repeating numbers
# This also increases the space we have to list more seeds
methods     = [ 'random' ]  # 'np', 'tf' ]  # just use random
cache_steps = 50
cache_seeds = int(4.25 * 100_000_000 / len(methods) / cache_steps)
cache = {
    method: np.load(f'{method}.npy')
    for method in methods
    if os.path.exists(f'{method}.npy')
}

def get_rng_sequence(seed, length, method='random', use_cache=True) -> List[int]:
    # Use cached results to avoid interfering with opponents RNG
    # Avoid using the RNG during runtime, to prevent affecting opponents RNG
    guess = []
    if ( use_cache
     and method in cache.keys()
     and seed   < cache[method].shape[0]
     and length < cache[method].shape[1]
    ):
        guess = cache[method][seed][:length]
    else:
        # If the results are not in the cache
        # then ensure we save and restore the random seed state to avoid affecting opponent's RNG
        if method == 'random':
            seed_state = random.getstate()
            random.seed(seed)
            guess = [ random.randint(0,2) for n in range(length) ]
            random.setstate(seed_state)
        elif method == 'np':
            seed_state = np.random.get_state()
            np.random.seed(seed)
            guess = np.random.randint(0,2, length)
            np.random.set_state(seed_state)
        elif method == 'tf':
            generator = tf.random.Generator.from_seed(seed)
            guess = generator.uniform((length,), minval=0, maxval=3, dtype=tf.dtypes.int32).numpy()
    return list(guess)


def random_seed_search(history: List[int], method='random', min_sequence=5) -> Tuple[int, int]:
    """
    Perform the search as a vectorized operation inside numpy
    rather than function call + cast to list in a python loop
    """
    seed     = None
    expected = None
    if method in cache.keys():
        size  = np.min([len(history), cache[method].shape[1]])
        seeds = np.where( np.all(
            cache[method][:,:size] == history[:size]
        , axis=1))[0]
        if len(seeds) >= 2:
            # Pick the first matching seed, and play out the full sequence
            seed = np.min(seeds)
            if cache[method].shape[1] > len(history):
                # Lookup the next number from the cache
                expected = cache[method][seed,len(history)]

                # NOTE: the most common early-game case is a hash collision
                #       alternately we could compute stats
                #       but the list of seeds will exponentially decrease as history gets longer
                # expected = np.argmax(np.bincount(cache[method][seeds,len(history)]))
            else:
                # Compute the remainder of the Mersenne Twister sequence
                expected = get_rng_sequence(seed, length=len(history) + 1, method=method, use_cache=False)[-1]
    return seed, expected




history       = []
solutions     = []
best_solution = None
min_seed      = 0

def win_symbol(opponent_action: int) -> str:
    return (
        ' ' if len(history) == 0               else
        '+' if opponent_action == (history[-1] + 0) % 3 else
        '|' if opponent_action == (history[-1] - 1) % 3 else
        '-'
    )


# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def random_seed_search_agent(observation, configuration, warmup=0, seeds_per_turn=200_000):
    # print(observation)
    global history, solutions, best_solution, min_seed
    safety_time   = 0.1
    time_start    = time.perf_counter()
    time_end      = time_start + configuration.actTimeout - safety_time
    opponent_action = observation.lastOpponentAction if observation.step > 0 else None
    if opponent_action is not None:
        history.append(opponent_action)

    action = None
    try:
        # Play the first few rounds as pure random to see what the opponent does
        if observation.step <= warmup:
            return irrational_agent(observation, configuration)


        # Search through list of irrational sequences
        for irrational_name, irrational in irrationals.items():
            for offset in [0,1,2]:    # Check for off by one sequences
                if irrational[:len(history)-offset] == history[offset:]:
                    prediction = irrational[len(history)+1]
                    action     = (prediction + 1) % configuration.signs
                    print(f'Found Irrational: {irrational_name} | {opponent_action}{win_symbol(opponent_action)} > action {prediction} ->  {action}')
                    return int(action)


        # # If we have found a seed, but gone beyong the cache limit, then short-circuit
        # if best_solution is not None:
        #     if observation.step > cache_steps:
        #         seed, method, offset, spin = best_solution
        #         guess      = get_rng_sequence(seed=seed, length=len(history) + 1 - offset, method=method, use_cache=False)
        #         guess      = [ (n + spin) % configuration.signs for n in guess ]
        #         prediction = guess[-1]
        #         action     = (prediction + 1) % configuration.signs
        #         print(f'Post Cache Seed: {best_solution} | {opponent_action}{win_symbol(opponent_action)} > action = {prediction} -> {action} | {solutions}')
        #         return int(action)


        # Search through the list of previously found solutions and see if any still match
        for seed, method, offset, spin in solutions:
            guess      = get_rng_sequence(length=len(history) + 1 - offset, seed=seed, method=method)[offset:]
            guess      = [ (n + spin) % configuration.signs for n in guess ]
            prediction = guess[-1]

            # BUG: when seed gets reused, opponent seems to skip a step
            if guess[:-1] == history:
                action        = (prediction + 1) % configuration.signs
                best_solution = (seed, method, offset, spin)
                print(f'Reused Seed: {best_solution} | {opponent_action}{win_symbol(opponent_action)} > action = {prediction} -> {action} | {solutions}')
                return int(action)
        else:
            # BUG: not finding seed 42 | TODO: debug
            # Perform the search as a vectorized operation inside the numpy loop
            for method in methods:
                seed, expected = random_seed_search(history, method)
                min_seed       = cache_seeds
                if expected is not None:
                    prediction = expected
                    action     = (prediction + 1) % configuration.signs
                    offset     = 0
                    spin       = 0
                    solution   = (seed, method, offset, spin)
                    solutions += [ solution ]

                    print(f'Found Seed: {seed} | {opponent_action}{win_symbol(opponent_action)} > action = {prediction} -> {action} | {solutions}')
                    return int(action)


            # # This is the old slow way of doing the search
            # # Continue search for seeds until timeout
            # # loop_count = int( seeds_per_turn / len(methods) / len(history) )
            # loop_count  = cache_seeds
            # spin        = 0  # disable slows things down
            # max_offset  = 0  # disable slows things down
            # max_history = min( len(history), cache_steps )
            # for seed in range(min_seed, min_seed + loop_count):
            #     for method in methods:
            #         guess      = get_rng_sequence(length=len(history)+1, seed=seed, method=method)
            #         prediction = guess[-1]
            #         for offset in range(0, max_offset+1):  # Check for off by one sequences
            #             if guess[:max_history-offset] == history[offset:offset+max_history]:
            #                 solution   = (seed, method, offset, spin)
            #                 solutions += [ solution ]
            #                 action     = (prediction + 1) % configuration.signs
            #                 print(f'Found Seed: {best_solution} | {opponent_action}{win_symbol(opponent_action)} > action = {prediction} -> {action}')
            #                 return int(action)
            #                 break
            # min_seed += 1
            #
            # # Only loop over the seeds within the cache
            # # This avoids touching the opponents RNG at runtime
            # # The cache is sufficently large that this will take a while
            # if seed >= cache_seeds:
            #     min_seed = 0
            #     break
            # if time.perf_counter() > time_end: break

    except Exception as exception:
        print(exception)

    if action is None:
        action = irrational_agent(observation, configuration, method='pi+1')


    time_taken = time.perf_counter() - time_start
    print(f'{observation.step:4d} | {time_taken:.6f}s | {opponent_action}{win_symbol(opponent_action)} > action = {action} | min_seed = {min_seed} | solutions = {solutions}')
    return int(action) % configuration.signs
