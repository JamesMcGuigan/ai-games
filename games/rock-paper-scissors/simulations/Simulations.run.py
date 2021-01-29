#!/usr/bin/env python3

import contextlib
import pprint
import time

from humanize import precisedelta
from joblib import delayed, Parallel
from kaggle_environments import evaluate

from simulations.Simulations import Simulations

time_start = time.perf_counter()
opponents = {
    # 'random':        random_agent,
    # 'rock':          rock_agent,
    # 'paper':         paper_agent,
    # 'scissors':      scissors_agent,
    # 'pi':            pi_agent,
    # 'anti_pi':       anti_pi_agent,
    # 'de_bruijn':     de_bruijn_sequence,
    # 'sequential':    sequential_agent,
    # 'reactionary':   reactionary_agent,
    # 'counter_react': counter_reactionary,
    # 'mirror':        mirror_opponent_agent,
    # 'statistical':   statistical,
    # 'anti_rotn':       anti_rotn_2,
    # 'tree':          decision_tree_agent,
    # 'iocaine':       iocaine_agent,
    # 'greenberg':     greenberg_agent,
    # 'stat_pred':     statistical_prediction_agent,
    # 'naive_bayes':   naive_bayes_agent,
}
opponents = Simulations.agents if not len(opponents) else opponents


def silent_agent(agent):
    def inner_agent(obs, conf):
        with contextlib.redirect_stdout(None):  # disable stdout for child agents
            return agent(obs, conf)
    return inner_agent if callable(agent) else agent

def simulate(agent_name, agent):
    print('#'*10 + f' {agent_name} ' + '#'*10)
    simulations_agent = Simulations()
    result = evaluate(
        "rps",
        [simulations_agent, silent_agent(agent)],
        configuration={
            "episodeSteps": 100,
            "actTimeout":   1000,
        },
        num_episodes=1,
        debug=True
    )
    return agent_name, result[0]

results = dict(Parallel(1)(
    delayed(simulate)(agent_name, agent)
    for agent_name, agent in opponents.items()
))
pp = pprint.PrettyPrinter(width=80, compact=True)
pp.pprint(results)

time_taken = time.perf_counter() - time_start
print(precisedelta(time_taken))
