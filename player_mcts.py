import argparse
import atexit
import math
import os
import pickle
import random
import time
from collections import defaultdict, namedtuple
from typing import List

from isolation import Agent, logger
from isolation.isolation import Action, Isolation
from run_match import TEST_AGENTS
from run_match_sync import play_sync
from sample_players import BasePlayer



MCTSRecord = namedtuple("MCTSRecord", ("wins","count","score"), defaults=(0,0,0))

class MCTSPlayer(BasePlayer):
    exploration = math.sqrt(2)
    data        = defaultdict(MCTSRecord)
    game        = Isolation

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
            with open("data.pickle", "rb") as file:
                cls.data.update({ **pickle.load(file), **cls.data })
                print("loaded: './data.pickle' ({:.2f}) Mb".format(os.path.getsize('./data.pickle')/1024/1024) )
                pass
        except (IOError, TypeError) as exception: pass

    @classmethod
    def save( cls ):
        # cls.load()  # update any new information from the file
        print("saving: './data.pickle'" )
        with open("data.pickle", "wb") as file:
            pickle.dump(cls.data, file)
        print("wrote:  './data.pickle' ({:.2f}) Mb)".format(os.path.getsize('./data.pickle')/1024/1024) )

    @classmethod
    def reset( cls ):
        cls.data = defaultdict(MCTSRecord)
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

        w = max(0, cls.data[child].wins)
        n = max(1, cls.data[child].count)   # avoid divide by zero
        N = max(2, cls.data[parent].count)  # encourage exploration of unexplored states | log(1) = 0
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
            record = MCTSRecord(
                wins  = cls.data[child].wins  + win,
                count = cls.data[child].count + 1,
                score = cls.score(child, parent)
            )
            cls.data[child] = record
            parent = child


def train_mcts(args):
    atexit.register(MCTSPlayer.save)

    agent1 = Agent(MCTSPlayer, "MCTSPlayer1")
    agent2 = TEST_AGENTS.get(args['opponent'].upper(), Agent(MCTSPlayer, "MCTSPlayer2"))
    agents = (agent1, agent2)

    history = {
        agent: []
        for agent in agents
    }
    start_time = time.perf_counter()
    match_id = 0
    while True:
        if args.get('rounds',  0) and args['rounds']  * 2 <= match_id:                         break
        if args.get('timeout', 0) and args['timeout']     <= time.perf_counter() - start_time: break

        match_id += 1
        agent_order = ( agents[(match_id)%2], agents[(match_id+1)%2] )  # reverse player order between matches
        winner, game_history, match_id = play_sync(agent_order, match_id=match_id, debug=True)

        winner_idx = agent_order.index(winner)
        loser      = agent_order[int(not winner_idx)]
        history[winner] += [ 1 ]
        history[loser]  += [ 0 ]

        MCTSPlayer.backpropagate(winner_idx, game_history)

        print('+' if winner == agent1 else '-', end='', flush=True)
        logn = 100
        if match_id and match_id % logn == 0:
            message = " match_id: {:4d} | last {} = {:0.0f}%".format(match_id, logn, 100 * sum(history[agent1][-logn:]) / logn )
            print(message); logger.info(message)

    MCTSPlayer.save()
    atexit.unregister(MCTSPlayer.save)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds',   type=int, default=0)
    parser.add_argument('-t', '--timeout',  type=int, default=0)
    parser.add_argument('-o', '--opponent', type=str, default='GREEDY')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    train_mcts(args)
