import random

random.seed(42)
def random_agent_seeded(observation, configuration):
    action = random.randint(0, configuration.signs-1)
    print(f'random_agent_seeded() = {action}')
    return int(action)
