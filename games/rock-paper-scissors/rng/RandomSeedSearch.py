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

# BUGFIX: Kaggle Submission Environment os.getcwd() == "/kaggle/working/"
if os.environ.get('GFOOTBALL_DATA_DIR', ''):
    os.chdir('/kaggle_simulations/agent/')


class RandomSeedSearch(IrrationalSearchAgent):
    """
    The Random Seed Search algorithm takes the default RNG `random.random(seed)`,
    and generates RNG sequences (length 20) for as many seeds as can fit in the
    100Mb submission filesize limit. This happens to be 20 million seeds.
    Numpy.size = `(20,000,000, 20) x int8` = 425.0 MB = 94Mb tar.gz.
    Compression offers a ~4.5x space saving,
    mostly due to the first 6 bits in every int8 being zero for trinary numbers.

    Pre-caching costs 27 minutes of compile time.
    By careful use of excluding previously rejected seeds,
    we can search this array very quickly and even implement
    offset search for the opponents history without the 1s timelimit.
    Each turn we can reduce the remaining search space by a factor of 3.
    This is compared to the previous implemention which could
    only search about ~10,000 seeds per turn using a python loop.

    If a seed is found, the next number in the sequence can be predicted
    without violating apparent randomness.
    This effectively acts as an opening book against Mersenne Twister RNG agents.
    What is observed in practice is the Random Seed Search is only occasionally able to
    find a small sequences numbers, often during the other agent's random warmup period.
    I have not yet seen a +500 score against an unseeded random agent.
    I suspect these partial matching sequences represent hash collisions
    of repeating sequences within the Mersenne Twister number.

    As the history gets longer, hash collisions become exponentially rarer.
    This is the game-winning difference between using a repeating number and irrational number
    for your source of randomness.
    The effect is quite small, but the 50% standard deviation for a random draw
    is in this game is only 20. A statistical 2-3 point advantage shifts
    the probability distribution of endgame score, a -21 score gets converted into a draw
    and a +19 score gets converted into a win.

    This is equivalent to the statistical effect that causes
    Black Holes to slowly lose mass via Hawking Radiation.

    Achievement Unlocked: Beat the unbeatable Nash Equilibrium RNG bot!
    """

    methods     = [ 'random' ]  # [ 'np', 'tf' ]  # just use random
    cache_steps = 20  # seeds are rarely found after move 15
    cache_seeds = int(4 * 100_000_000 / len(methods) / cache_steps)  # 100Mb x 4.25x tar.gz compression
    cache = {
        method: np.load(f'{method}.npy')
        for method in methods
        if os.path.exists(f'{method}.npy')
    }

    def __init__(self, min_length=4, max_offset=1000, use_stats=True, cheat=False, verbose=True):
        """
        :param min_length:  minimum sequence length for a match 3^4 == 1/81 probability
        :param max_offset:  maximum offset to search for from start of history
        :param use_stats:   if True pick the most probable continuation rather than minimum seed
        :param cheat:       set the opponents seed - only works on localhost
        :param verbose:     log output to console
        """
        self.use_stats  = use_stats
        self.cheat      = cheat   # needs to be set before super()
        self.min_length = min_length
        self.max_offset = max_offset
        self.conf       = {'episodeSteps': 1000, 'actTimeout': 1000, 'agentTimeout': 15, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
        super().__init__(verbose=verbose)
        self.print_cache_size()

    def reset(self):
        """ This gets called at the beginning of every game """
        if self.verbose >= 2: print(f'{self.__class__.__name__} | reset()')
        super().reset()

        # SPEC: self.candidate_seeds[method][offset] = list(seeds)
        self.candidate_seeds = defaultdict(lambda: defaultdict(dict))
        # SPEC: self.repeating_seeds[method] = count
        self.repeating_seeds = defaultdict(lambda: defaultdict(int))
        if self.cheat:
            self.set_global_seed()


    # obs  {'step': 168, 'lastOpponentAction': 0}
    # conf {'episodeSteps': 1000, 'actTimeout': 1000, 'agentTimeout': 15, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
    def action(self, obs, conf):
        # NOTE: self.history state is managed in the parent class
        self.conf = conf

        # This allows testing of the agent without actually submitting to the competition
        if obs.step > 900:
            raise Exception("Don't Submit To Competition")

        # This is a short circuit to speed up unit tests
        irrational, irrational_name = self.search_irrationals(self.history['opponent'])
        if irrational is not None and obs.step > self.min_length + 2:
            return super().action(obs, conf)

        # Search the Random Seed Cache
        # Important to do this each turn as it reduces self.candidate_seeds[offset] by 1/3
        expected, seed, offset, method = self.search_cache(self.history['opponent'])

        # If have multiple or zero seed matches, but also an irrational, then use that
        if seed is None and irrational is not None:
            action = super().action(obs, conf)

        elif expected is not None:
            action   = (expected + 1) % conf.signs
            opponent = ( self.history['opponent'] or [None] )[-1]
            if seed is None: seed = f"{'many':>12s}"
            else:            seed = f"{seed + (offset or 0)/1000:12.3f}"
            if self.verbose:
                print(
                    f"{obs.step:4d} | {opponent}{self.win_symbol()} > action {expected} -> {action} | " +
                    f"Found RNG Seed: {method:6s} {seed} |",
                    self.log_repeating_seeds()
                )

        else:
            # Default to parent class to return a secure Irrational sequence
            action = super().action(obs, conf)

        return int(action) % conf.signs



    ### Searching

    def search_cache(self, history: List[int]) \
            -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        """
        Search though the range of different pre-cached numpy arrays for a seed sequence match
        """
        expected, seed, offset, method = None, None, None, None
        for method in self.methods:
            seed, expected, offset = self.search_cache_method(history, method=method)
            if expected is not None:
                break
        return expected, seed, offset, method


    def search_cache_method(self, history: List[int], method='random') \
            -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Perform a vectorized numpy search for the opponent's RNG sequence
        This allows 8.5 million records to be searched in about 0.2ms
        Compared to a search rate of about ~10,000/s using list() comparison in a python loop
        """
        time_start = time.perf_counter()
        seed     = None
        expected = None
        offset   = None
        if method in self.cache.keys() and len(history):

            # Keep track of sequences we have already excluded, to improve performance
            seeds, offset = self.find_candidate_seeds(history, method)
            sequence      = history[offset:]

            if self.use_stats and len(seeds) >= 2:
                # the most common early-game case is a hash collision
                # we can compute stats on the distribution of next numbers
                # the list of matching seeds will exponentially decrease as history gets longer
                seed     = None
                stats    = np.bincount(self.cache[method][seeds,len(sequence)])
                expected = np.argmax(stats)
                if np.count_nonzero(stats) == 1:  # all seeds agree
                    seed = np.min(seeds)

            elif len(seeds):
                # Pick the first matching seed, and play out the full sequence
                seed = np.min(seeds)

                if self.cache[method].shape[1] > len(sequence):
                    # Lookup the next number from the cache
                    expected = self.cache[method][seed,len(sequence)]
                else:
                    # Compute the remainder of the Mersenne Twister sequence
                    expected = self.get_rng_sequence(seed, length=len(sequence) + 1, method=method)[-1]

            # This is a log of how much statistical advantage we gained
            if seed is not None:
                self.repeating_seeds[method][seed + offset/1000] += 1

        if self.verbose >= 2:
            time_taken = time.perf_counter() - time_start
            print(f'{self.__class__.__name__} | search_cache({method}): {time_taken*1000:.3f}ms')

        return seed, expected, offset


    def find_candidate_seeds(self, history, method: str, timeout=0.5) -> Tuple[np.ndarray, int]:
        """
        Find a list of candidate seeds for a given sequence
        This makes searching through the cache very fast
        We can effectively exclude a third of the cache on ever iteration

        This now also implements offset search to find more candidates
        Despite using a loop here, the search time is somewhat stable around 600ms
        As we can exclude 1/3 of the dataset for each offset at each turn

        By implementing offset search we constantly find seeds throughout the match
        and RandomSeedSearch stops being just an opening book
        """

        seeds_by_offset = {}
        max_offset = min(self.max_offset, len(history))
        for offset in range(max_offset):
            sequence = history[offset:]
            size     = np.min([len(sequence), self.cache[method].shape[1]])
            if size < self.min_length: continue  # reduce the noise of matching sequences

            # Have we already searched for this offset and excluded possibilities
            if offset in self.candidate_seeds[method]:
                candidates = self.candidate_seeds[method][offset]
                if len(candidates) == 0:
                    seeds = candidates  # we found an empty list
                else:
                    seeds_idx  = np.where( np.all(
                        self.cache[method][ candidates, :size ] == sequence[ :size ]
                    , axis=1))[0]
                    seeds = candidates[seeds_idx]
            else:
                seeds = np.where( np.all(
                    self.cache[method][ : , :size ] == sequence[ :size ]
                , axis=1))[0]

            self.candidate_seeds[method][offset] = seeds
            if len(seeds):
                seeds_by_offset[offset] = seeds

        # Return the search that returned the shortest list
        if len(seeds_by_offset) == 0:
            seeds  = np.array([])
            offset = 0
        else:
            offset = sorted(seeds_by_offset.items(), key=lambda pair: len(pair[1]))[0][0]
            seeds  = seeds_by_offset[offset]
        return seeds, offset


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

    def log_repeating_seeds(self) -> dict:
        """
        Format self.repeating_seeds for logging

        3^4 = 1 in 81   chance, which is the default minimum sequence length
        3^5 = 1 in 243  chance
        3^6 = 1 in 729  chance, which in theory should happen once per game
        3^7 = 1 in 2187 chance, which is a statistical advantage
        """
        repeating_seeds = {}
        for method in self.repeating_seeds.keys():
            repeating_seeds[method] = {
                key: value
                for key, value in self.repeating_seeds[method].items()
                if value >= self.min_length
            }
            repeating_seeds[method] = dict(sorted(repeating_seeds[method].items(), key=itemgetter(1), reverse=True))
            if len(repeating_seeds[method]) == 0:
                del repeating_seeds[method]
        return repeating_seeds


    def print_cache_size(self):
        """ Mostly for debugging purposes, log the contents of the cache """
        if self.verbose:
            filenames = [ (filename + ' = ' + naturalsize(os.path.getsize(filename))) for filename in glob.glob('*') ]
            print('tar.gz =', filenames)
            print(f'{self.__class__.__name__}::cache.keys()', list(self.cache.keys()))
            for name, cache in self.cache.items():
                print(f'{self.__class__.__name__}::cache[{name}] = {cache.shape}' )


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
