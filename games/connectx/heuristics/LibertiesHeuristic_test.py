import math
from typing import Type

import pytest
from kaggle_environments import make

from core.ConnectX import ConnectX
from core.Heuristic import Heuristic
from heuristics.LibertiesHeuristic import LibertiesHeuristic



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
    return LibertiesHeuristic

@pytest.fixture
def game(observation, configuration, heuristic_class) -> ConnectX:
    return ConnectX(observation, configuration, heuristic_class)



def test_first_move(game):
    results = [ game.result(action) for action in game.actions ]
    scores  = [ result.heuristic.score for result in results ]
    best_score, best_action = max(zip(scores, game.actions))
    assert best_action == 3

def test_second_move(game):
    game    = game.result(3)
    results = [ game.result(action) for action in game.actions ]
    scores  = [ result.heuristic.score for result in results ]
    best_score, best_action = max(zip(scores, game.actions))
    assert best_action == 3


def test_gameover(game):
    for i in range(3):
        game = game.result(3).result(0)
        assert game.gameover == False

    game = game.result(3)
    assert game.gameover == True
    assert game.score    == math.inf
    assert game.utility  == math.inf
