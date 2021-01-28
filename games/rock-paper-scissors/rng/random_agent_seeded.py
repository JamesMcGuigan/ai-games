import random

random.seed(42)
def random_agent_seeded(observation, configuration):
    return random.randint(0, configuration.signs-1)
