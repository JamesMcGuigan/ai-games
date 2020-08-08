#!/usr/bin/env python3
import itertools

from kaggle_environments import make
from kaggle_environments.envs.connectx.connectx import negamax_agent
from kaggle_environments.envs.connectx.connectx import random_agent

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.AntColonyTreeSearch import AntColonyTreeSearch
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloLinkedList import MontyCarloLinkedList
from agents.Negamax.Negamax import Negamax

training_agents = [
    # AntColonyTreeSearch(),
    # MontyCarloLinkedList(),
    MontyCarloHeuristic(),
]
opponent_agents = [
    random_agent,
    negamax_agent,
    AlphaBetaAgent.agent(),
    AlphaBetaBitboard.agent(),
    AntColonyTreeSearch(),
    MontyCarloLinkedList(),
    MontyCarloHeuristic(),
    Negamax(),
    *[ Negamax(max_depth=max_depth) for max_depth in range(1,8+1) ],
]
# random.shuffle(training_agents)
# random.shuffle(opponent_agents)


env = make("connectx", debug=True)
env.render()
env.reset()

def print_results(env, agent_order):
    try:
        if   env.state[0].reward == env.state[1].reward:  print(f'Draw: {agent_order[0].__name__} vs {agent_order[1].__name__}')
        elif env.state[0].reward  > env.state[1].reward:  print(f'Winner player 1: {agent_order[0].__name__} | Loser player 2: {agent_order[1].__name__}')
        else:                                             print(f'Winner player 2: {agent_order[1].__name__} | Loser player 1: {agent_order[0].__name__}')
    except:
        print(f'Error: {agent_order[0].__name__} vs {agent_order[1].__name__}')
    print()


safety_time = 2
for timeout in [3,6 ]:
    timeout += safety_time
    env.configuration.timeout = timeout  # MontyCarlo has a safety time of 2s, so this gives 1s of runtime to expand nodes

    # # Next play 100 rounds against random agent (7x7x2 = 98) which should cover most 2-deep opening positions
    # for agent_1 in training_agents:
    #     agent_2 = random_agent
    #     for round in range(100):
    #         agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
    #         env.run(agent_order)
    #         print_results(env, agent_order)
    #         env.reset()
    #
    # # Have each training agent play against each other
    # for agent_1, agent_2 in itertools.product(training_agents, training_agents):
    #     for round in range(2):
    #         agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
    #         env.run(agent_order)
    #         print_results(env, agent_order)
    #         env.reset()

    # Train against real opponents
    for agent_1, agent_2 in itertools.product(training_agents, opponent_agents):
        for round in range(2):
            agent_order = [agent_1, agent_2] if round % 2 == 0 else [agent_2, agent_1]
            env.run(agent_order)
            print_results(env, agent_order)
            env.reset()