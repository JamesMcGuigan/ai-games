import argparse
import atexit
import math
import os
import pickle
import random
import time
from collections import namedtuple
from typing import List

from isolation import Agent, logger
from isolation.isolation import Action, Isolation
from player_alphabeta import AlphaBetaAreaPlayer, AlphaBetaPlayer
from run_match_sync import play_sync
from sample_players import BasePlayer, GreedyPlayer, MinimaxPlayer, RandomPlayer



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
        # action, score = max(sorted(zip(actions, scores), key=itemgetter(1)))
        action = self.choose_with_probability(actions, scores)
        self.queue.put(action)
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
    exploration = math.sqrt(2)  # use math.sqrt(2) for training, and 0 for playing

class MCTSPlayer(MCTS):
    exploration = 0  # use math.sqrt(2) for training, and 0 for playing

    @classmethod
    def backpropagate( cls, agent_idx, winner_idx: int, game_history: List[Action] ):
        return None


def train_mcts(args):
    if args.get('save', 1):
        atexit.register(MCTS.save)  # Autosave on Ctrl-C

    agent1 = TEST_AGENTS.get(args['agent'].upper(),    Agent(MCTSTrainer, "MCTSTrainer"))
    agent2 = TEST_AGENTS.get(args['opponent'].upper(), Agent(MCTSTrainer, "MCTSTrainer"))
    if agent1.name == agent2.name:
        agent1 = Agent(agent1.agent_class, agent1.name+'1')
        agent2 = Agent(agent2.agent_class, agent2.name+'2')
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
        winner, game_history, match_id = play_sync(agent_order, match_id=match_id)

        winner_idx = agent_order.index(winner)
        loser      = agent_order[int(not winner_idx)]
        scores[winner] += [ 1 ]
        scores[loser]  += [ 0 ]

        for agent_idx, agent in enumerate(agent_order):
            if callable(getattr(agent.agent_class, 'backpropagate', None)):
                agent.agent_class.backpropagate(agent_idx, winner_idx, game_history)

        if args.get('progress'):
            print('+' if winner == agents[0] else '-', end='', flush=True)

        frequency = args.get('frequency', 100)
        if (  frequency != 0 and match_id % frequency == 0
           or match_id  != 0 and match_id == args.get('rounds')
        ):
            message = " match_id: {:4d} | last {} = {:3.0f}% | all = {:3.0f}% | {} vs {}" .format(
                match_id, frequency,
                100 * sum(scores[agents[0]][-frequency:]) / frequency,
                100 * sum(scores[agents[0]]) / len(scores[agents[0]]),
                agents[0].name,
                agents[1].name,
            )
            print(message); logger.info(message)

    if args.get('save', 1):
        MCTS.save()
        atexit.unregister(MCTS.save)


TEST_AGENTS = {
    "RANDOM":    Agent(RandomPlayer,        "Random Agent"),
    "GREEDY":    Agent(GreedyPlayer,        "Greedy Agent"),
    "MINIMAX":   Agent(MinimaxPlayer,       "Minimax Agent"),
    "ALPHABETA": Agent(AlphaBetaPlayer,     "AlphaBeta Agent"),
    "AREA":      Agent(AlphaBetaAreaPlayer, "AlphaBeta Area Agent"),
    "MCA":       Agent(MCTSPlayer,          "MCTS Agent"),
    "MCT":       Agent(MCTSTrainer,         "MCTS Trainer"),
}
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds',     type=int, default=0)
    parser.add_argument(      '--timeout',    type=int, default=0)    # train_mcts() timeout global for
    parser.add_argument('-t', '--time_limit', type=int, default=150)  # play_sync()  timeout per round
    parser.add_argument('-a', '--agent',      type=str, default='MCT')
    parser.add_argument('-o', '--opponent',   type=str, default='MCA')
    parser.add_argument('-f', '--frequency',  type=int, default=1000)
    parser.add_argument(      '--progress',   action='store_true')    # show progress bat
    parser.add_argument('-s', '--save',       type=int, default=1)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    train_mcts(args)
