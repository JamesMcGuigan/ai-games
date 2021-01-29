import pprint
import time

from humanize import precisedelta
from kaggle_environments import evaluate

from simulations.Simulations import Simulations

time_start = time.perf_counter()
results    = {}
opponents  = Simulations.agents
# opponents  = {
#     'random':        random_agent,
#     'statistical':   statistical,
#     'stat_pred':     statistical_prediction_agent,
#     'naive_bayes':   naive_bayes_agent,
# }

for agent_name, agent in opponents.items():
    simulations_agent = Simulations()
    result = evaluate(
        "rps",
        [simulations_agent, agent],
        configuration={
            "episodeSteps": 100,
            # "actTimeout":   1000,
        },
        num_episodes=1,
        debug=False
    )
    results[agent_name] = result[0]

pp = pprint.PrettyPrinter(width=80, compact=True)
pp.pprint(results)

time_taken = time.perf_counter() - time_start
print(precisedelta(time_taken))
