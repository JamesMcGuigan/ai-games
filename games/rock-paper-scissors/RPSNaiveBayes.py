# %%writefile submission.py
from collections import defaultdict
import random
from typing import *

import numpy as np
from pydash import flatten


class RPSNaiveBayes():
    def __init__(self, max_memory=100, verbose=True):
        self.max_memory = max_memory
        self.verbose    = verbose
        self.history = {
            "opponent": [],
            "expected": [],
            "action":   [],
        }
        self.memory = {
            "action":          defaultdict(lambda: np.array([0,0,0])),
            "opponent":        defaultdict(lambda: np.array([0,0,0])),
            "action,opponent": defaultdict(lambda: np.array([0,0,0])),
        }

    def __call__(self, obs, conf):
        return self.agent(obs, conf)


    # obs  {'remainingOverageTime': 60, 'step': 1, 'reward': 0, 'lastOpponentAction': 0}
    # conf {'episodeSteps': 10, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3, 'tieRewardThreshold': 20, 'agentTimeout': 60}
    def agent(self, obs, conf):
        # print('obs', obs)
        self.update_state(obs, conf)

        views          = self.get_current_views()
        log_likelihood = self.get_log_likelihood(views)
        probability    = np.exp(log_likelihood) / np.sum(np.exp(log_likelihood))

        expected = random.choices( population=[0,1,2], weights=probability, k=1 )[0]
        action   = int(expected + 1) % conf.signs
        self.history['expected'].insert(0, expected)
        self.history['action'].insert(0, action)

        if self.verbose:
            print(f'step = {obs.step:4d} | action = {action} | expected = {expected} | probability', probability.round(3), 'log_likelihood', log_likelihood.round(3))

        return int(action)


    def update_state(self, obs, conf):
        if obs.step > 0:
            self.history['opponent'].insert(0, obs.lastOpponentAction % conf.signs)

        for keys in self.memory.keys():
            memories = self.get_new_memories(keys)
            for value, path in memories:
                self.memory[keys][path][value] += 1


    def get_key_min_length(self, keys: str) -> int:
        min_length = min([ len(self.history[key]) for key in keys.split(',') ])
        return min_length


    def get_new_memories(self, keys: Union[str,List[str]]) -> List[Tuple[Tuple,int]]:
        min_length = self.get_key_min_length(keys)
        min_length = min(min_length, self.max_memory)
        memories   = []
        for n in range(1,min_length):
            value = self.history["opponent"][0]
            paths = []
            for key in keys.split(','):
                path = self.history[key][1:n]
                if len(path): paths.append(path)
            paths = tuple(flatten(paths))
            if len(paths):
                memories.append( (value, paths) )
        return memories


    def get_current_views(self) -> Dict[str, List[Tuple[int]]]:
        views = {
            keys: [
                tuple(flatten([value, paths]))
                for (value, paths) in self.get_new_memories(keys)
            ]
            for keys in self.memory.keys()
        }
        return views


    def get_log_likelihood(self, views: List[Tuple]) -> np.ndarray:
        log_likelihoods = np.array([.0,.0,.0])
        for keys in self.memory.keys():
            count = np.sum( np.array(list(self.memory[keys].values())).shape )
            for path in views[keys]:
                try:
                    n_unique = 3 ** len(path)
                    freqs = self.memory[keys][path]  # * n_unique  # TODO: do we need this?
                    probs = (freqs + 1) / ( count + n_unique )     # Laplacian Smoothing
                    # probs = freqs / count
                    log_likelihood = [
                        np.log( probs[a] / (probs[b] + probs[c]) )
                        if (probs[b] + probs[c]) > 0 else 0.0
                        for a, b, c in [ (0,1,2), (1,2,0), (2,0,1) ]
                    ]
                    log_likelihoods += np.array(log_likelihood)
                except ZeroDivisionError: pass

        return log_likelihoods


instance = RPSNaiveBayes()
def kaggle_agent(obs, conf):
    return instance.agent(obs, conf)
