# DOCS: https://classroom.udacity.com/nanodegrees/nd898/parts/bc65db39-9306-46f8-be4b-061c76f89108/modules/c921eb41-5e97-4b75-838f-95dfae3c3108/lessons/e3d2343d-a663-407a-9bbb-9a00504f7826/concepts/1ecef15d-82f5-4078-8f52-8d6424ee2a28
import random
import sys
import time
from operator import itemgetter

from agents.MCTS import MCTSMaximum
from isolation import Agent
from run_match_sync import play_sync
from sample_players import BasePlayer



class UCTPlayer(BasePlayer):
    """ UCTPlayer """
    agent = MCTSMaximum
    # agent = MCTSRandom
    verbose_depth = False

    def __init__(self, time_limit=150, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_limit = time_limit
        MCTSMaximum.load()

    def get_action(self, state):
        start_time = time.perf_counter()
        actions = state.actions()
        self.queue.put( random.choice(actions) )
        states  = [ state.result(action) for action in actions ]
        agents  = ( Agent(self.agent, '1'), Agent(MCTSMaximum, '2') )

        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)
        for index in range(sys.maxsize):
            try:
                winner, game_history, match_id = play_sync(agents, state, time_limit=0, logging=False)
                winner_idx = agents.index(winner)
                self.agent.backpropagate(winner_idx, game_history)
                scores        = [ self.agent.data[state].score for state in states if state in self.agent.data ]
                action, score = max( zip(actions,scores), key=itemgetter(1) )
                self.queue.put(action)
            except: pass
            if self.verbose_depth: print(index, end=' ', flush=True)
            if (time.perf_counter() - start_time) > (self.time_limit / 1024): break



