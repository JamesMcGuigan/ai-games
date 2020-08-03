#!/usr/bin/env python3
import argparse
import traceback

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.MontyCarloTreeSearch import MontyCarloTreeSearch
from agents.Negamax.Negamax import Negamax



env = make("connectx", debug=True)
env.render()
env.reset()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',        action="store_true")
parser.add_argument('--inline',       action="store_true")
parser.add_argument('-r', '--rounds', type=int, default=1)
parser.add_argument('-1', '--p1',     type=str, required=True)
parser.add_argument('-2', '--p2',     type=str, default='negamax')
argv = parser.parse_args()


agent_args = {}
if argv.debug:
    env.configuration.timeout = 24*60*60
    env.configuration.steps   = 1

agent_1 = agent_2 = agent_1_name = agent_2_name = None
for agent_name, position in [ (argv.p1, 'p1'), (argv.p2, 'p2') ]:
    if   agent_name == 'AlphaBetaAgent':       agent = AlphaBetaAgent.agent()
    elif agent_name == 'AlphaBetaBitboard':    agent = AlphaBetaBitboard.agent()
    elif agent_name == 'MontyCarloTreeSearch': agent = MontyCarloTreeSearch
    elif agent_name == 'Negamax':              agent = Negamax()
    elif agent_name == 'negamax':              agent = agent_name
    elif agent_name == 'random':               agent = agent_name
    else: raise Exception(f'runner.py: invalid agent {position} == {agent_name}')

    if position == 'p1':
        agent_1 = agent
        agent_1_name = agent_name
    else:
        agent_2 = agent
        agent_2_name = agent_name

if argv.inline:
    # env.configuration.timeout = 120
    observation   = env.state[0].observation
    configuration = env.configuration
    agent_1(observation, configuration)
    # agent_2(observation, configuration)

else:
    scores = [0,0]
    rounds = 0
    try:
        for round in range(argv.rounds):
            rounds    += 1
            env.reset()
            env.run([agent_1, agent_2])
            # noinspection PyTypeChecker
            env.render(mode="human")
            scores[0] += ((env.state[0].reward or 0) + 1)/2
            scores[1] += ((env.state[1].reward or 0) + 1)/2
    except Exception as exception:
        print('runner.py: Exception: ', exception)
        traceback.print_tb(exception.__traceback__)

    print()
    print('runner.py', argv)
    print(f'{scores[0]:3.1f}/{rounds:.1f} = {scores[0]/rounds:.2f} | {agent_1_name}')
    print(f'{scores[1]:3.1f}/{rounds:.1f} = {scores[1]/rounds:.2f} | {agent_2_name}')
    if scores[0] == scores[1]:
        print('Draw!')
    else:
        winner = agent_1_name if scores[0] < scores[1] else agent_2_name
        print(f'Winner: {winner}')
