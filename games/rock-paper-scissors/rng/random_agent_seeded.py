import random

def random_agent_seeded(observation, configuration, seed=42):
    # Set a deterministic seed
    if observation.step == 0:
        random.seed(seed)

    action = random.randint(0, configuration.signs-1)
    print(f'random_agent_seeded({seed}) = {action}')
    return int(action)
