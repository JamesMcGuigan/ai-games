#!/usr/bin/env python3
import itertools
import random

from kaggle_environments import make
from kaggle_environments.envs.connectx.connectx import negamax_agent, random_agent

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.AntColonyTreeSearch import AntColonyTreeSearch
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloLinkedList import MontyCarloLinkedList
from agents.Negamax.Negamax import Negamax
from agents.RulesAgent import CenterBot

training_agents = [
    AntColonyTreeSearch(),
    MontyCarloLinkedList(),
    MontyCarloHeuristic(),
]
opponent_agents = [
    Negamax(),
    AlphaBetaAgent.agent(),
    AlphaBetaBitboard.agent(),
    CenterBot,
    negamax_agent,
    random_agent,
]
random.shuffle(training_agents)
random.shuffle(opponent_agents)


env = make("connectx", debug=True)
env.render()
env.reset()

# First have each training agent play against each other
for agent_1, agent_2 in itertools.product(training_agents, training_agents):
    for round in range(2):
        agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
        env.run(agent_order)
        env.reset()

# Next play 100 rounds against random agent (7x7x2 = 98) which should cover most 2-deep opening positions
for agent_1 in training_agents:
    agent_2 = random_agent
    for round in range(100):
        agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
        env.run(agent_order)
        env.reset()

# Train against real opponents
for agent_1, agent_2 in itertools.product(training_agents, training_agents):
    for round in range(2):
        agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
        env.run(agent_order)
        env.reset()

# Repeat self-training after updated weights
for agent_1, agent_2 in itertools.product(training_agents, training_agents):
    for round in range(2):
        agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
        env.run(agent_order)
        env.reset()