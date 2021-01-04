import random

def random_agent(observation, configuration):
    random.seed(None)
    return random.randint(0, configuration.signs-1)
