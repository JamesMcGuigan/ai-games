# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-memory-patterns

import random
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple

import numpy as np


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class MemoryPatterns:
    def __init__(self, min_memory=2, max_memory=20, threshold=0.5, warmup=5, verbose=True):
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.threshold  = threshold
        self.warmup     = warmup
        self.verbose    = verbose
        self.history = {
            "step":      [],
            "reward":    [],
            "opponent":  [],
            "pattern":   [],
            "action":    [],
            # "rotn_self": [],
            # "rotn_opp":  [],
        }
        self.obs  = None
        self.conf = None


    def __call__(self, obs, conf):
        return self.agent(obs, conf)


    # obs  {'remainingOverageTime': 60, 'step': 1, 'reward': 0, 'lastOpponentAction': 0}
    # conf {'episodeSteps': 1000, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3, 'tieRewardThreshold': 20, 'agentTimeout': 60}
    def agent(self, obs, conf):
        # print('obs', obs)
        # print('conf', conf)
        self.obs  = obs
        self.conf = conf
        self.update_state(obs, conf)
        if obs.step < self.warmup:
            expected = self.random_action(obs, conf)
        else:
            for keys in [ ("opponent", "action"), ("opponent",) ]:
                # history  = self.generate_history(["opponent", "action"])  # "action" must be last
                history  = self.generate_history(["opponent"])
                memories = self.build_memory(history)
                patterns = self.find_patterns(history, memories)
                if len(patterns): break
            score, expected, pattern = self.find_best_pattern(patterns)
            self.history['pattern'].append(pattern)
            if self.verbose:
                print('keys    ', keys)
                print('history ', history)
                print('memories', memories)
                print('patterns', patterns)
                print('score   ', score)
                print('expected', expected)
                print('pattern ', pattern)

        action = (expected + 1) % conf.signs
        self.history['action'].append(action)

        if self.verbose:
            print('action', action)
        return int(action)


    def random_action(self, obs, conf) -> int:
        return random.randint(0, conf.signs-1)

    def sequential_action(self, obs, conf) -> int:
        return (obs.step + 1) % conf.signs


    def update_state(self, obs, conf):
        self.history['step'].append( obs.step )
        self.history['reward'].append( obs.reward )
        if obs.step != 0:
            self.history['opponent'].append( obs.lastOpponentAction )
            # rotn_self = (self.history['opponent'][-1] - self.history['opponent'][-2]) % conf.signs 
            # rotn_opp  = (self.history['opponent'][-1] - self.history['action'][-1]))  % conf.signs
            # self.history['rotn_self'].append( rotn_self )
            # self.history['rotn_opp'].append( rotn_opp )


    def generate_history(self, keys: List[str]) -> List[Tuple[int]]:
        # Reverse order to correctly match up arrays
        history = list(zip(*[ reversed(self.history[key]) for key in keys ]))
        history = list(reversed(history))
        return history


    def build_memory(self, history: List[Tuple[int]]) -> List[ Dict[Tuple[int], List[int]] ]:
        output    = [ dict() ] * self.min_memory
        expecteds = self.generate_history(["opponent"])
        for batch_size in range(self.min_memory, self.max_memory+1):
            if batch_size >= len(history): break  # ignore batch sizes larger than history
            output_batch    = defaultdict(lambda: [0,0,0])
            history_batches  = list(batch(history, batch_size+1))
            expected_batches = list(batch(expecteds, batch_size+1))
            for n, (pattern, expected_batch) in enumerate(zip(history_batches, expected_batches)):
                previous_pattern = tuple(pattern[:-1])
                expected         = (expected_batch[-1][-1] or 0) % self.conf.signs  # assume "action" is always last 
                output_batch[ previous_pattern ][ expected ] += 1
            output.append( dict(output_batch) )
        return output


    def find_patterns(self, history: List[Tuple[int]], memories: List[ Dict[Tuple[int], List[int]] ]) -> List[Tuple[float, int, Tuple[int]]]:
        patterns = []
        for n in range(1, self.max_memory+1):
            if n >= len(history): break

            pattern = tuple(history[-n:])
            if pattern in memories[n]:
                score    = np.std(memories[n][pattern])
                expected = np.argmax(memories[n][pattern])
                patterns.append( (score, expected, pattern) )
        patterns = sorted(patterns, key=itemgetter(0), reverse=True)
        return patterns


    def find_best_pattern(self, patterns: List[Tuple[float, int, Tuple[int]]] ) -> Tuple[float, int, Tuple[int]]:
        patterns       = sorted(patterns, key=itemgetter(0), reverse=True)
        pattern_scores = self.get_pattern_scores()
        for (score, expected, pattern) in patterns:
            break
            # if pattern in pattern_scores:
            #     if pattern_scores[pattern] > self.threshold:
            #         break
            #     else:
            #         expected += 1
            #         break
            # else:
            #     break
        else:
            score    = 0.0
            expected = self.random_action(self.obs, self.conf)
            pattern  = tuple()
        return score, expected, pattern


    def get_pattern_scores(self):
        pattern_rewards = defaultdict(list)
        for reward, pattern in self.generate_history(["reward", "pattern"]):
            pattern_rewards[pattern].append( reward )
        pattern_scores = { pattern: np.mean(rewards) for patten, rewards in pattern_rewards.items() }
        return pattern_scores



memory_patterns_instance = MemoryPatterns()
def memory_patterns_agent(obs, conf):
    return memory_patterns_instance(obs, conf)
