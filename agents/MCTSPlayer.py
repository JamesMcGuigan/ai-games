import math
import os
import pickle
import random
import time
from collections import namedtuple
from operator import itemgetter
from typing import List

from isolation.isolation import Action, Isolation
from sample_players import BasePlayer



MCTSRecord = namedtuple("MCTSRecord", ("wins","count","score"), defaults=(0,0,0))

class MCTS(BasePlayer):
    exploration = math.sqrt(2)  # use math.sqrt(2) for training, and 0 for playing
    game = Isolation
    file = './data.pickle'
    data = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load()

    # def __del__(self):
    #     self.save()

    @classmethod
    def load( cls ):
        if cls.data: return  # skip loading if the file is already in class memory
        try:
            # class data may be more upto date than the pickle file, so avoid race conditions with multiple instances
            start_time = time.perf_counter()
            with open(cls.file, "rb") as file:
                # print("loading: "+cls.file )
                cls.data.update({ **pickle.load(file), **cls.data })
                print("loaded: "+cls.file+" | {:.1f}MB in {:.1f}s".format(
                    os.path.getsize(cls.file)/1024/1024,
                    time.perf_counter() - start_time
                ))
        except (IOError, TypeError, EOFError) as exception:
                pass

    @classmethod
    def save( cls ):
        # cls.load()  # update any new information from the file
        if cls.data:
            # print("saving: " + cls.file )
            with open(cls.file, "wb") as file:
                start_time = time.perf_counter()
                pickle.dump(cls.data, file)
                print("wrote:  "+cls.file+" | {:.1f}MB in {:.1f}s".format(
                    os.path.getsize(cls.file)/1024/1024,
                    time.perf_counter() - start_time
                ))

    @classmethod
    def reset( cls ):
        cls.data = {}
        cls.save()


    def get_action(self, state):
        actions  = state.actions()
        children = [ state.result(action) for action in actions ]
        scores   = [ self.score(child, state) for child in children ]
        action   = self.choose(actions, scores)
        self.queue.put(action)
        return action

    def choose( self, actions: List[Action], scores: List[int] ) -> Action:
        return self.choose_with_probability(actions, scores)

    @staticmethod
    def choose_maximum( actions: List[Action], scores: List[int] ) -> Action:
        action, score = max(sorted(zip(actions, scores), key=itemgetter(1)))
        return action

    @staticmethod
    def choose_with_probability( actions: List[Action], scores: List[int] ) -> Action:
        # Efficient method to compute a weighted probability lookup without division
        # Source: https://www.kaggle.com/jamesmcguigan/ant-colony-optimization-algorithm
        total  = sum(scores)
        rand   = random.random() * total
        action = None
        for (weight, action) in zip(scores, actions):
            if rand > weight: rand -= weight
            else:             break
        return action


    # noinspection PyArgumentList,PyTypeChecker
    @classmethod
    def score( cls, child, parent ):
        # DOCS: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        # Chang, Fu, Hu, and Marcus. Kocsis and Szepesvári recommend w/n + c√(ln N/n)
        # wi stands for the number of wins for the node considered after the i-th move
        # ni stands for the number of simulations for the node considered after the i-th move
        # Ni stands for the total number of simulations after the i-th move run by the parent node of the one considered
        # c is the exploration parameter—theoretically equal to √2; in practice usually chosen empirically

        # Avoid using defaultdict, as it creates too many lookup entries with zero score
        child_record  = cls.data[child]  if child  in cls.data else MCTSRecord()
        parent_record = cls.data[parent] if parent in cls.data else MCTSRecord()
        w = max(0, child_record.wins)
        n = max(1, child_record.count)   # avoid divide by zero
        N = max(2, parent_record.count)  # encourage exploration of unexplored states | log(1) = 0
        score = w/n + cls.exploration * math.sqrt(math.log(N)/n)
        return score


    # noinspection PyTypeChecker, PyArgumentList
    @classmethod
    def backpropagate( cls, agent_idx, winner_idx: int, game_history: List[Action] ):
        winner_idx = winner_idx % 2
        parent = cls.game()
        idx    = -1
        for action in game_history:
            idx   = (idx + 1) % 2
            win   = int(idx == winner_idx)
            child = parent.result(action)

            if agent_idx == idx:   # only learn from the agent's moves
                # Avoid using defaultdict, as it creates too many lookup entries with zero score
                child_record = cls.data[child] if child in cls.data else MCTSRecord()
                record = MCTSRecord(
                    wins  = child_record.wins  + win,
                    count = child_record.count + 1,
                    score = cls.score(child, parent)
                )
                cls.data[child] = record

            parent = child


class MCTSTrainer(MCTS):
    data = {}
    file = './MCTSTrainer.pickle'
    exploration = 0               # use math.sqrt(2) for training, and 0 for playing
    # exploration = math.sqrt(2)  # use math.sqrt(2) for training, and 0 for playing
    def choose( self, actions: List[Action], scores: List[int] ) -> Action:
        return self.choose_with_probability(actions, scores)

class MCTSPlayer(MCTS):
    data = {}
    file = './MCTSPlayer.pickle'
    # file = './MCTSTrainer.pickle'
    exploration = math.sqrt(2)

    def choose( self, actions: List[Action], scores: List[int] ) -> Action:
        return self.choose_maximum(actions, scores)
