# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-weighted-random-agent

import random

opponent_frequency = [1,1,1]
def weighted_random_agent(observation, configuration, exponent=0):
    if observation.step > 0:
        opponent_frequency[observation.lastOpponentAction] += (observation.step ** exponent)

    expected_action = random.choices( population=[0,1,2], weights=opponent_frequency, k=1 )[0]
    counter_action  = ( expected_action + 1 ) % configuration.signs
    return counter_action
