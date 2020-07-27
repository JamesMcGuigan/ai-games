#!/usr/bin/env python3
import argparse

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.MontyCarloTreeSearch import MontyCarloTreeSearch
from agents.MontyCarlo.MontyCarloTreeSearch import precompile_numba



precompile_numba()
env = make("connectx", debug=True)
env.render()
env.reset()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',  action="store_true")
parser.add_argument('--inline', action="store_true")
parser.add_argument('--vs',     type=str)
argv = parser.parse_args()


agent_args = {}
if argv.debug:
    env.configuration.timeout = 24*60*60
    env.configuration.steps   = 1

if argv.inline:
    # env.configuration.timeout = 120
    observation   = env.state[0].observation
    configuration = env.configuration
    MontyCarloTreeSearch(observation, configuration)
else:
    agent    = MontyCarloTreeSearch
    opponent = MontyCarloTreeSearch
    if argv.vs:
        if   argv.vs == 'AlphaBetaAgent':       opponent = AlphaBetaAgent.agent()
        elif argv.vs == 'AlphaBetaBitboard':    opponent = AlphaBetaBitboard.agent()
        elif argv.vs == 'MontyCarloTreeSearch': opponent = MontyCarloTreeSearch
        else:                                   opponent = argv.vs

    env.run([agent, opponent])
    # noinspection PyTypeChecker
    env.render(mode="human")

