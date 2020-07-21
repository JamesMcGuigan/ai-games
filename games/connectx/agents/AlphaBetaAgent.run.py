#!/usr/bin/env python3
from kaggle_environments import make

from agents.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgentOdd import AlphaBetaAgentOdd



env = make("connectx", debug=True)
env.render()
env.reset()
env.configuration.timeout = 24*60*60
env.configuration.steps   = 1

# env.run([AlphaBetaAgent.agent, "random"])
env.run([ AlphaBetaAgent.agent(search_max_depth=5), AlphaBetaAgentOdd.agent(search_max_depth=5) ])
# noinspection PyTypeChecker
env.render(mode="human")
