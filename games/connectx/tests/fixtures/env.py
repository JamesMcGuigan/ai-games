import pytest
from kaggle_environments import make



@pytest.fixture
def env():
    env = make("connectx", debug=True)
    env.configuration.timeout = 4.25  # Reduce timer for quicker tests + 0.25s safety time
    return env

# observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
# configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
@pytest.fixture
def configuration(env):
    configuration = env.configuration
    return configuration


@pytest.fixture
def observation(env):
    observation = env.state[0].observation
    return observation
