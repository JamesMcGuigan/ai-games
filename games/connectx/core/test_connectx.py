import pytest
from kaggle_environments import make

from games.connectx.core.ConnectX import ConnectX



@pytest.fixture
def env():
    env = make("connectx", debug=True)
    env.configuration.timeout = 24*60*60
    return env

@pytest.fixture
def observation(env):
    return env.state[0].observation

@pytest.fixture
def configuration(env):
    return env.configuration

@pytest.fixture
def game(observation, configuration) -> ConnectX:
    return ConnectX(observation, configuration)


def test_actions_returns_list_of_int(game: ConnectX):
    assert isinstance(game.actions, list)
    assert all( isinstance(action, int) for action in game.actions )
