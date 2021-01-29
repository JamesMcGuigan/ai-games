# White Belt Agents from https://www.kaggle.com/chankhavu/rps-dojo
import random

from kaggle_environments.envs.rps.utils import get_score

last_counter_action = None
def counter_reactionary(observation, configuration):
    global last_counter_action
    if observation.step == 0:
        last_counter_action = random.randrange(0, configuration.signs)
    elif get_score(last_counter_action, observation.lastOpponentAction) == 1:
        last_counter_action = (last_counter_action + 2) % configuration.signs
    else:
        last_counter_action = (observation.lastOpponentAction + 1) % configuration.signs
    return last_counter_action
