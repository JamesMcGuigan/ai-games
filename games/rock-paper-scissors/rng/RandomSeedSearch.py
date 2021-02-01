# %%writefile -a main.py
# %%writefile RandomSeedSearch.py
# Source: https://www.kaggle.com/jamesmcguigan/random-seed-search-nash-equilibrium-opening-book/
# Source: https://github.com/JamesMcGuigan/ai-games/blob/master/games/rock-paper-scissors/rng/RandomSeedSearch.py

import glob
import os
import random
import time
from collections import defaultdict
from operator import itemgetter
from typing import List, Optional, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable tensorflow logging

import numpy as np
import tensorflow as tf
from humanize import naturalsize

from rng.IrrationalSearchAgent import IrrationalSearchAgent


class RandomSeedSearch(IrrationalSearchAgent):
    """
    The Random Seed Search algorithm takes the default RNG random.random(seed),
    and generates RNG sequences (length 50) for as many seeds as can fit in the 100Mb submission filesize limit.
    This happens to be 8,500,000 seeds. Numpy.size = (8500000, 50)int8 = 425.0 MB = 94Mb tar.gz.
    Compression offers a ~4.5x space saving, mostly due to the first 6 bits in every int8 being zero for trinary numbers.
    Pre-caching costs 23 minutes of compile time, but at runtime a numpy vectorized search takes only 0.2ms
    for everything compared to ~10,000 seeds per second searching using a python loop.

    If a seed is found, the next number in the sequence can be predicted without violating apparent randomness.
    This effectively acts as an opening book against Mersenne Twister RNG agents.
    What is observed in practice is the Random Seed Search is only occasionally
    able to find a small sequence of 2-3 numbers, often during the other agent's random warmup period.
    I have not yet seen a +500 score against an unseeded random agent.

    I suspect these partial matching sequences represent hash collisions
    of repeating sequences within the Mersenne Twister number.
    As the history gets longer, hash collisions become exponentially rarer.
    This is the game-winning difference between using a repeating number vs and irrational number
    for your source of randomness. The effect is quite small,
    but the standard deviation for a random draw is in this game is only 20.
    A statistical 2-3 point advantage shifts the probability distribution of endgame score,
    a -21 score gets converted into a draw and a +19 score gets converted into a win.
    This is equivalent to the statistical effect that causes Black Holes to slowly lose mass via Hawking Radiation.

    Achievement Unlocked: Beat the unbeatable Nash Equilibrium RNG bot!
    """

    methods     = [ 'random' ]  # [ 'np', 'tf' ]  # just use random
    cache_steps = 25  # seeds are rarely found after move 15
    cache_seeds = int(4.25 * 100_000_000 / len(methods) / cache_steps)  # 100Mb x 4.25x tar.gz compression
    cache = {
        method: np.load(f'{method}.npy')
        for method in methods
        if os.path.exists(f'{method}.npy')
    }

    def __init__(self, use_stats=False, verbose=True, cheat=False):
        self.use_stats = use_stats
        self.cheat     = cheat             # needs to be set before super()
        super().__init__(verbose=verbose)

    def reset(self):
        """ This gets called at the beginning of every game """
        if self.verbose >= 2: print(f'{self.__class__.__name__} | reset()')
        super().reset()
        self.min_seed        = { name: 0 for name in self.cache.keys() }  # must start at zero
        self.repeating_seeds = defaultdict(lambda: defaultdict(int))      # count the number repeating seed sequences
        self.no_more_seeds   = False             # short circuit if we reach the end of the cache

        if self.cheat:
            self.set_global_seed()

        if self.verbose:
            # Log the cache size
            filenames = [ (filename + ' = ' + naturalsize(os.path.getsize(filename))) for filename in glob.glob('*') ]
            print('tar.gz =', filenames)
            print(f'{self.__class__.__name__}::cache.keys()', self.cache.keys())
            for name, cache in self.cache.items():
                print(f'{self.__class__.__name__}::cache[{name}] = {cache.shape}' )


    def action(self, obs, conf):
        # self.history state is managed in the parent class

        # Check for irrationals first, and if so use parent class for logging
        # This also speeds up unit tests
        expected, irrational_name = self.search_irrationals(self.history['opponent'])
        if expected is not None:
            action = super().action(obs, conf)
        else:
            # Search the Random Seed Cache
            expected, seed, method = self.search_cache(self.history['opponent'])
            if expected is not None:
                action   = (expected + 1) % conf.signs
                opponent = ( self.history['opponent'] or [None] )[-1]
                if self.verbose:
                    print(
                        f"{obs.step:4d} | {opponent}{self.win_symbol()} > action {expected} -> {action} | " +
                        f"Found RNG Seed: {method} {seed:8d} |",
                        self.log_repeating_seeds()
                    )
            else:
                # Default to parent class to return a secure Irrational sequence
                action = super().action(obs, conf)

        return int(action) % conf.signs



    ### Searching

    def search_cache(self, history: List[int]) -> Tuple[int, int, str]:
        """
        Search though the range of different pre-cached numpy arrays for a seed sequence match
        """
        expected, seed, method = None, None, None
        for method in self.methods:
            seed, expected = self.search_cache_method(history, method=method)
            if expected is not None:
                break
        return expected, seed, method


    def search_cache_method(self, history: List[int], method='random') -> Tuple[Optional[int], Optional[int]]:
        """
        Perform a vectorized numpy search for the opponent's RNG sequence
        This allows 8.5 million records to be searched in about 0.2ms
        Compared to a search rate of about ~10,000/s using list() comparison in a python loop
        """
        time_start = time.perf_counter()
        seed     = None
        expected = None
        if method in self.cache.keys() and len(history) and not self.no_more_seeds:
            min_seed = self.min_seed[method]
            size  = np.min([len(history), self.cache[method].shape[1]])
            seeds = np.where( np.all(
                self.cache[method][ min_seed: , :size ] == history[ :size ]
            , axis=1))[0] + min_seed

            if self.use_stats and len(seeds) >= 2:
                # the most common early-game case is a hash collision
                # we can compute stats on the distribution of next numbers
                # the list of matching seeds will exponentially decrease as history gets longer
                seed     = None
                stats    = np.bincount(self.cache[method][seeds,len(history)])
                expected = np.argmax(stats)

            elif len(seeds):
                # Pick the first matching seed, and play out the full sequence
                # Also log the seed to reduce search time during subsequent turns
                seed = np.min(seeds)
                if seed == self.min_seed[method]:
                    # This is a log of how much statistical advantage we gained
                    self.repeating_seeds[method][seed] += 1
                self.min_seed[method] = seed

                if self.cache[method].shape[1] > len(history):
                    # Lookup the next number from the cache
                    expected = self.cache[method][seed,len(history)]
                else:
                    # Compute the remainder of the Mersenne Twister sequence
                    expected = self.get_rng_sequence(seed, length=len(history) + 1, method=method)[-1]

            else:
                # Shot circuit for the rest of the game if we didn't find anything
                # Search takes upto 800ms, so don't waste overage time during the endgame
                self.no_more_seeds = True

        if self.verbose >= 2:
            time_taken = time.perf_counter() - time_start
            print(f'{self.__class__.__name__} | search_cache({method}): {time_taken*1000:.3f}ms')

        return seed, expected


    def get_rng_sequence(self, seed, length, method='random', use_cache=True) -> List[int]:
        """
        Generates the first N numbers for a given Mersenne Twister seed sequence

        Results may potentially be cached, though careful attention is paid to
        save and restore the internal state of random.random() to prevent us
        from affecting the opponent's RNG sequence, and accidentally stealing numbers from their sequence
        """
        sequence = []
        if ( use_cache
         and method in self.cache.keys()
         and seed   < self.cache[method].shape[0]
         and length < self.cache[method].shape[1]
        ):
            sequence = self.cache[method][seed][:length]
        else:
            # If the results are not in the cache
            # then ensure we save and restore the random seed state to avoid affecting opponent's RNG
            if method == 'random':
                seed_state = random.getstate()
                random.seed(seed)
                sequence = [ random.randint(0,2) for _ in range(length) ]
                random.setstate(seed_state)
            elif method == 'np':
                seed_state = np.random.get_state()
                np.random.seed(seed)
                sequence = np.random.randint(0,2, length)
                np.random.set_state(seed_state)
            elif method == 'tf':
                generator = tf.random.Generator.from_seed(seed)
                sequence = generator.uniform((length,), minval=0, maxval=3, dtype=tf.dtypes.int32).numpy()
        return list(sequence)



    ### Precaching

    def precache(self) -> List[str]:
        # BUGFIX: Kaggle joblib PicklingError: Could not pickle the task to send it to the workers
        return [ self.precache_method(method) for method in self.methods ]

    def precache_method(self, method='random') -> str:
        """
        Compute all the random.random() Mersenne Twister RNG sequences
        for the first 17 million seeds.
        Only the first 25 numbers from each sequence are returned,
        but we rarely find cache hits or hash collisions after the first 15 moves.

        These numbers are configurable via cls.cache_steps and cls.cache_seeds

        This takes about 23 minutes of runtime and generates 425Mb of int8 data
        This compresses to 94Mb in tar.gz format which is under the 100Mb submission limit
        """
        filename = f'{method}.npy'
        shape    = (self.cache_seeds, self.cache_steps)
        if os.path.exists(filename) and np.load(filename).shape == shape:
            print(f'cached {filename:10s} =', shape, '=', naturalsize(os.path.getsize(filename)))
        else:
            method_cache = np.array([
                self.get_rng_sequence(seed=seed, length=self.cache_steps, method=method)
                for seed in range(self.cache_seeds)
            ], dtype=np.int8)
            np.save(filename, method_cache)
            print(f'wrote  {filename:10s} =', method_cache.shape, '=', naturalsize(os.path.getsize(filename)))
        return filename



    ### Logging

    def log_repeating_seeds(self, min_value=3):
        """
        Format self.repeating_seeds for logging

        "Once is happenstance. Twice is coincidence. The third time it's enemy action"
        A sequence that has repeats once  is a 1/3 chance, so ignore
        A sequence that has repeats twice is a 1/9 chance, so ignore
        A sequence that has repeats three is a 1/27 chance, is a statistical advantage!
        """
        repeating_seeds = {}
        for method in self.repeating_seeds.keys():
            repeating_seeds[method] = {
                key: value
                for key, value in self.repeating_seeds[method].items()
                if value >= min_value
            }
            repeating_seeds[method] = dict(sorted(repeating_seeds[method].items(), key=itemgetter(1), reverse=True))
            if len(repeating_seeds[method]) == 0:
                del repeating_seeds[method]
        return repeating_seeds



    ### Cheats

    @staticmethod
    def set_global_seed(seed=42):
        """
        This is a cheat to set the opponent's RNG seed
        It only works when playing on localhost or inside a Kaggle Notebook
        This is due to both agents playing within the same python process
        and thus sharing random.random() RNG state

        This doesn't work inside the kaggle leaderboard submission environment
        as both agents play within different CPU processes.

        This attack also assumes that we do not ourselves invoke the
        random.random() RNG ourselves at runtime, which is solved by
        precaching the first 8.5 million seed sequences at compile time
        """
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)



random_seed_search_instance = RandomSeedSearch()
def random_seed_search_agent(obs, conf):
    return random_seed_search_instance.agent(obs, conf)
