#!/usr/bin/env python3
import argparse

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.AntColony.AntColonyTreeSearch import AntColonyTreeSearch

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
    AntColonyTreeSearch(observation, configuration)
else:
    agent    = AntColonyTreeSearch
    opponent = AntColonyTreeSearch
    if argv.vs:
        if   argv.vs == 'AlphaBetaAgent':    opponent = AlphaBetaAgent.agent()
        elif argv.vs == 'AlphaBetaBitboard': opponent = AlphaBetaBitboard.agent()
        else:                                opponent = argv.vs

    env.run([agent, opponent])
    # noinspection PyTypeChecker
    env.render(mode="human")

