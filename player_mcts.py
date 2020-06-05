import argparse
import atexit
import math
import os
import pickle
import random
import time
from collections import defaultdict, namedtuple
from operator import itemgetter
from typing import Dict, List, Tuple

from isolation import Agent, logger
from isolation.isolation import Action, Isolation
from player_alphabeta import AlphaBetaPlayer
from run_match_sync import play_sync
from sample_players import BasePlayer, GreedyPlayer, MinimaxPlayer, RandomPlayer



MCTSRecord = namedtuple("MCTSRecord", ("wins","count","score"), defaults=(0,0,0))

class MCTSPlayer(BasePlayer):
    exploration = 0   # use math.sqrt(2) for training, and 0 for playing
    game = Isolation
    file = 'data.pickle'
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
            with open(cls.file, "rb") as file:
                cls.data.update({ **pickle.load(file), **cls.data })
                print("loaded: './data.pickle' ({:.2f}) MB".format(os.path.getsize(cls.file)/1024/1024) )
                pass
        except (IOError, TypeError) as exception: pass

    @classmethod
    def save( cls ):
        # cls.load()  # update any new information from the file
        print("saving: " + cls.file )
        with open(cls.file, "wb") as file:
            pickle.dump(cls.data, file)
        # for key, value in cls.data.items():
        #     print(key, ':', value)
        print("wrote:  "+cls.file+" ({:.2f}) MB)".format(os.path.getsize(cls.file)/1024/1024) )

    @classmethod
    def reset( cls ):
        cls.data = defaultdict(MCTSRecord())
        cls.save()


    def get_action(self, state):
        actions  = state.actions()
        children = [ state.result(action) for action in actions ]
        scores   = [ self.score(child, state) for child in children ]
        # action, score = max(sorted(zip(actions, scores), key=itemgetter(1)))
        action = self.choose_with_probability(actions, scores)
        self.queue.put(action)
        return action


    @staticmethod
    def choose_with_probability( actions: List[Action], scores: List[int] ) -> Action:
        # Efficient method to compute a weighted probability lookup without division
        # Source: https://www.kaggle.com/jamesmcguigan/ant-colony-optimization-algorithm
        total = sum(scores)
        rand  = random.random() * total
        for (weight, action) in zip(scores, actions):
            if rand > weight: rand -= weight
            else:             break
        return action


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


    @classmethod
    def backpropagate( cls, winner_idx: int, game_history: List[Action] ):
        winner_idx = winner_idx % 2
        parent = cls.game()
        idx    = -1
        for action in game_history:
            idx   = (idx + 1) % 2
            win   = int(idx == winner_idx)
            child = parent.result(action)

            # Avoid using defaultdict, as it creates too many lookup entries with zero score
            child_record = cls.data[child] if child in cls.data else MCTSRecord()
            record = MCTSRecord(
                wins  = child_record.wins  + win,
                count = child_record.count + 1,
                score = cls.score(child, parent)
            )
            cls.data[child] = record
            parent = child


class MCTSTrainer(MCTSPlayer):
    exploration = math.sqrt(2)  # use math.sqrt(2) for training, and 0 for playing



def train_mcts(args):
    atexit.register(MCTSPlayer.save)

    agent1 = Agent(MCTSTrainer, "MCTSTrainer")
    agent2 = TEST_AGENTS.get(args['opponent'].upper(), Agent(MCTSPlayer, "MCTSPlayer"))
    agents = (agent1, agent2)

    scores = {
        agent: []
        for agent in agents
    }
    start_time = time.perf_counter()
    match_id = 0
    while True:
        if args.get('rounds',  0) and args['rounds']  <= match_id:                         break
        if args.get('timeout', 0) and args['timeout'] <= time.perf_counter() - start_time: break

        match_id += 1
        agent_order = ( agents[(match_id)%2], agents[(match_id+1)%2] )  # reverse player order between matches
        winner, game_history, match_id = play_sync(agent_order, match_id=match_id, debug=True)

        winner_idx = agent_order.index(winner)
        loser      = agent_order[int(not winner_idx)]
        scores[winner] += [ 1 ]
        scores[loser]  += [ 0 ]

        MCTSPlayer.backpropagate(winner_idx, game_history)

        print('+' if winner == agents[0] else '-', end='', flush=True)
        line = 100 # args['verbose']
        if match_id and match_id % line == 0:
            message = " match_id: {:4d} | last {} = {:3.0f}% | all = {:3.0f}% | {} vs {}" .format(
                match_id, line,
                100 * sum(scores[agents[0]][-line:]) / line,
                100 * sum(scores[agents[0]]) / len(scores[agents[0]]),
                agents[0].agent_class.__name__,
                agents[1].agent_class.__name__,
            )
            print(message); logger.info(message)

    MCTSPlayer.save()
    atexit.unregister(MCTSPlayer.save)


TEST_AGENTS = {
    "RANDOM":    Agent(RandomPlayer,    "Random Agent"),
    "GREEDY":    Agent(GreedyPlayer,    "Greedy Agent"),
    "MINIMAX":   Agent(MinimaxPlayer,   "Minimax Agent"),
    "ALPHABETA": Agent(AlphaBetaPlayer, "AlphaBeta Agent"),
    "MCTS":      Agent(MCTSPlayer,      "MCTS Agent"),
    "MCTST":     Agent(MCTSTrainer,     "MCTS Trainer"),
}
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds',   type=int, default=0)
    parser.add_argument('-t', '--timeout',  type=int, default=0)
    parser.add_argument('-a', '--agent',    type=str, default='MCTST')
    parser.add_argument('-o', '--opponent', type=str, default='MCTST')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    train_mcts(args)
