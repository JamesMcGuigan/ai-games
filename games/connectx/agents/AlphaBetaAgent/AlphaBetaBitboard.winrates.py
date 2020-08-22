#!/usr/bin/env python3
# seq 5 | parallel --ungroup --jobs 2 agents/AlphaBetaAgent/AlphaBetaBitboard.winrates.py -r 2 -t 5
import argparse
import multiprocessing
import time

from joblib import delayed
from joblib import Parallel
from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.AlphaBetaAgent.AlphaBetaOddEven import AlphaBetaOddEven
from heuristics.BitsquaresHeuristic import bitsquares_heuristic
from heuristics.OddEvenHeuristic import oddeven_bitsquares_heuristic

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rounds',  type=int, default=30)
parser.add_argument('-t', '--timeout', type=int, default=1 + 2)  # 1s for faster benchmarks | AlphaBetaBitboard has depth=4 opening
parser.add_argument('argv', nargs=argparse.REMAINDER)
argv = parser.parse_args()


env = make("connectx", debug=True)
env.configuration.timeout = argv.timeout   # Half normal timer for quicker tests
observation   = env.state[0].observation
configuration = env.configuration

rounds   = argv.rounds
agent    = AlphaBetaBitboard
opponent = AlphaBetaOddEven


scores = { agent.__name__: 0, opponent.__name__: 0 }
def run_round(agent, opponent, round=0):
    kwargs = {
        "verbose_depth":  False,
    }
    agent_kwargs    = {
        "heuristic_fn": bitsquares_heuristic(),
        **kwargs
    }
    opponent_kwargs = {
        "heuristic_fn": oddeven_bitsquares_heuristic(),
        **kwargs
    }

    env = make("connectx", debug=True)
    env.configuration.timeout = argv.timeout
    time_start = time.perf_counter()
    if round % 2 == 0:
        agent_order = [ agent.agent(**agent_kwargs), opponent.agent(**opponent_kwargs) ]
        env.run(agent_order)
        if   env.state[0].reward ==  1: scores[agent.__name__]    += 1
        elif env.state[0].reward == -1: scores[opponent.__name__] += 1
        else: scores[agent.__name__] += 0.5; scores[opponent.__name__] += 0.5;
    else:
        agent_order = [ opponent.agent(**opponent_kwargs), agent.agent(**agent_kwargs) ]
        env.run(agent_order)
        if   env.state[1].reward ==  1: scores[agent.__name__]    += 1
        elif env.state[1].reward == -1: scores[opponent.__name__] += 1
        else: scores[agent.__name__] += 0.5; scores[opponent.__name__] += 0.5;
    time_taken = time.perf_counter() - time_start
    print(f'{time_taken:6.1f}s', scores)
    return scores

n_jobs = rounds if rounds < multiprocessing.cpu_count() * 2 else multiprocessing.cpu_count()
all_scores = Parallel(n_jobs=n_jobs)([
    delayed(run_round)(agent, opponent, round)
    for round in range(rounds)
])

wins     = sum([ scores[agent.__name__]    for scores in all_scores ])
losses   = sum([ scores[opponent.__name__] for scores in all_scores ])
winrate  = 100 * wins   / rounds
lossrate = 100 * losses / rounds
print(f'{  wins:.1f}/{rounds:.0f} = { winrate:3.0f}% winrate  {agent.__name__} vs {opponent.__name__}')
print(f'{losses:.1f}/{rounds:.0f} = {lossrate:3.0f}% lossrate {agent.__name__} vs {opponent.__name__}')


### Winrates @ timeout=5
#  55% winrate AlphaBetaBitboard vs AlphaBetaAgent | original heuristic
#  70% winrate AlphaBetaBitboard vs AlphaBetaAgent | + log2() % 1 == 0
#  60% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=1   without math.log2() (mostly draws)
#  80% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=1   with math.log2() (all wins/losses)
#  80% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=2   with math.log2() (all wins/losses)
#  70% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=4   with math.log2() (all wins/losses)
#  80% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=8   with math.log2() (all wins/losses)
# 100% winrate AlphaBetaBitboard vs AlphaBetaAgent | + double_attack_score=0.5 with math.log2() (all wins/losses)
