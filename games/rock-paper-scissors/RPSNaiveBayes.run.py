#!/usr/bin/env python3
import sys

import numpy as np
from kaggle_environments import evaluate, make

from random_agent import random_agent
from RPSNaiveBayes import RPSNaiveBayes

agent1 = RPSNaiveBayes()
agent2 = random_agent

env = make("rps", configuration={"episodeSteps": 1000}, debug=True)
env.run([agent1, agent2])
# # env.run(["submission.py", '../input/rock-paper-scissors-xgboost/submission.py'])
# # env.run(["submission.py", 'submission.py'])
env.render(mode="ansi")

result = evaluate("rps", [ agent1, agent2 ], configuration={
    "agentTimeout": sys.maxsize,
    "actTimeout":   sys.maxsize
})
result = np.array(result).flatten()
print(result)
