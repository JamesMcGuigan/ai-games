#!/usr/bin/env python3
# seq 5 | parallel --ungroup --jobs 2 agents/AlphaBetaAgent/AlphaBetaAgentBitboard.winrates.py -r 2 -t 5
import argparse
import gc
import time

from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaAgentBitboard import AlphaBetaAgentBitboard

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rounds',  type=int, default=10)
parser.add_argument('-t', '--timeout', type=int, default=5)  # Half time for faster benchmarks
parser.add_argument('argv', nargs=argparse.REMAINDER)
argv = parser.parse_args()


env = make("connectx", debug=True)
env.configuration.timeout = 8   # Half normal timer for quicker tests
observation   = env.state[0].observation
configuration = env.configuration

rounds   = argv.rounds
agent    = AlphaBetaAgentBitboard
opponent = AlphaBetaAgent
kwargs = {
    "verbose_depth":    False,
}

scores = { agent.__name__: 0, opponent.__name__: 0 }
for round in range(rounds):
    gc.collect()
    env = make("connectx", debug=True)
    env.configuration.timeout = argv.timeout
    time_start = time.perf_counter()
    if round % 2 == 0:
        env.run([agent.agent(**kwargs), opponent.agent(**kwargs)])
        if   env.state[0].reward ==  1: scores[agent.__name__]    += 1
        elif env.state[0].reward == -1: scores[opponent.__name__] += 1
        else: scores[agent.__name__] += 0.5; scores[opponent.__name__] += 0.5;
    else:
        env.run([opponent.agent(**kwargs), agent.agent(**kwargs)])
        if   env.state[1].reward ==  1: scores[agent.__name__]    += 1
        elif env.state[1].reward == -1: scores[opponent.__name__] += 1
        else: scores[agent.__name__] += 0.5; scores[opponent.__name__] += 0.5;
    time_taken = time.perf_counter() - time_start
    print(f'{time_taken:6.2f}s', scores)

wins    = scores[agent.__name__]
winrate = 100 * wins / rounds
print(f'{wins:.1f}/{rounds:.0f} = {winrate:3.0f}% winrate {agent.__name__} vs {opponent.__name__}')

### Winrates @ timeout=5
#  55% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | original heuristic
#  70% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + log2() % 1 == 0
#  60% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=1   without math.log2() (mostly draws)
#  80% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=1   with math.log2() (all wins/losses)
#  80% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=2   with math.log2() (all wins/losses)
#  70% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=4   with math.log2() (all wins/losses)
#  80% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=8   with math.log2() (all wins/losses)
# 100% winrate AlphaBetaAgentBitboard vs AlphaBetaAgent | + double_attack_score=0.5 with math.log2() (all wins/losses)
