from typing import Type

import pytest
from kaggle_environments import make

from games.connectx.core.ConnectX import ConnectX
from games.connectx.core.Heuristic import Heuristic
from games.connectx.heuristics.LinesHeuristic import LinesHeuristic



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
def heuristic_class() -> Type[Heuristic]:
    return LinesHeuristic

@pytest.fixture
def game(observation, configuration, heuristic_class) -> ConnectX:
    return ConnectX(observation, configuration, heuristic_class)




def test_actions_returns_list_of_int(game: ConnectX):
    assert isinstance(game.actions, list)
    assert all( isinstance(action, int) for action in game.actions )


def test_game_result_actions_1(game):
    results = [ game.result(action).board[-1,:].tolist() for action in game.actions ]
    assert results == [
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1],
    ]

def test_game_result_actions_2(game):
    game = game.result(3)
    results = [ game.result(action).board[-1,:].tolist() for action in game.actions ]
    assert results == [
        [2,0,0,1,0,0,0],
        [0,2,0,1,0,0,0],
        [0,0,2,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,2,0,0],
        [0,0,0,1,0,2,0],
        [0,0,0,1,0,0,2],
    ]
