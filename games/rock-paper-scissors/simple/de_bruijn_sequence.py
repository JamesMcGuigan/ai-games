# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-de-bruijn-sequence

from itertools import combinations_with_replacement

import pydash

# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def de_bruijn_sequence(observation, configuration):
    actions = list(combinations_with_replacement([2,1,0,2,1,0],3)) * 18
    actions = pydash.flatten(actions)
    action  = actions[observation.step] % configuration.signs
    return int(action)
