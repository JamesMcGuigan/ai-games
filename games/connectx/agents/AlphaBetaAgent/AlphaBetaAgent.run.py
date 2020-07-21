#!/usr/bin/env python3
import argparse

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaAgentOdd import AlphaBetaAgentOdd



env = make("connectx", debug=True)
env.render()
env.reset()

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action="store_true")
argv = parser.parse_args()


if argv.debug:
    env.configuration.timeout = 24*60*60
    env.configuration.steps   = 1
    agent_args = {
        "search_max_depth": 5
    }
else:
    agent_args = {}


# env.run([AlphaBetaAgent.agent, "random"])
# env.run([ AlphaBetaAgent.agent(search_max_depth=5), AlphaBetaAgentOdd.agent(search_max_depth=5) ])
env.run([AlphaBetaAgent.agent(**agent_args), AlphaBetaAgentOdd.agent(**agent_args)])
# noinspection PyTypeChecker
env.render(mode="human")

