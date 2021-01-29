# White Belt Agents from https://www.kaggle.com/chankhavu/rps-dojo
import random

from kaggle_environments.envs.rps.utils import get_score

last_react_action = None
def reactionary_agent(observation, configuration):
    global last_react_action
    if observation.step == 0:
        last_react_action = random.randrange(0, configuration.signs)
    elif get_score(last_react_action, observation.lastOpponentAction) <= 1:
        last_react_action = (observation.lastOpponentAction + 1) % configuration.signs
    return last_react_action
