import random

random_agent_seed = 42
random.seed(random_agent_seed)
def random_agent_seeded(observation, configuration):
    action = random.randint(0, configuration.signs-1)
    print(f'random_agent_seeded({random_agent_seed}) = {action}')
    return int(action)
