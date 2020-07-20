import pytest
from kaggle_environments import make

from agents.RandomAgent import RandomAgent



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


def test_first_move(observation, configuration):
    action = RandomAgent.agent(observation, configuration)
    assert 0 <= action < 7
    assert type(action) == int

def test_can_play_game_against_self(env):
    env.run([RandomAgent.agent, RandomAgent.agent])
