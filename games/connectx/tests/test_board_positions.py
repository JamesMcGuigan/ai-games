import pytest
from kaggle_environments import make
from kaggle_environments.envs.connectx.connectx import play

from tests.fixtures.agents import agents



@pytest.mark.parametrize("agent_name, agent", agents)
def test_p1_one_move_to_win(observation, configuration, agent_name, agent):
    observation.mark  = 1
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,
        0,2,1,0,0,0,0,
        0,2,1,2,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action == 2



@pytest.mark.parametrize("agent_name, agent", agents)
def test_p2_one_move_to_lose(observation, configuration, agent_name, agent):
    observation.mark  = 2
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,
        0,0,1,0,0,0,0,
        0,2,1,2,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action == 2



@pytest.mark.parametrize("agent_name, agent", agents)
def test_setup_double_attack(observation, configuration, agent_name, agent):
    observation.mark  = 1
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,
        0,0,2,0,0,0,0,
        0,0,1,1,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action in [1, 4]


@pytest.mark.parametrize("agent_name, agent", agents)
def test_foil_double_attack(observation, configuration, agent_name, agent):
    observation.mark  = 2
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,
        0,0,1,1,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action in [0, 1, 4, 5]



@pytest.mark.parametrize("agent_name, agent", agents)
def test_foil_double_attack(observation, configuration, agent_name, agent):
    observation.mark  = 2
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,
        0,0,1,1,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action in [0, 1, 4, 5]


@pytest.mark.parametrize("agent_name, agent", agents)
def test_foil_double_attack(observation, configuration, agent_name, agent):
    observation.mark  = 2
    observation.board = [
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,
        0,0,1,1,0,0,0,
    ]
    action = agent(observation, configuration)
    assert action in [1, 4]

    play(observation.board, column=action, mark=2, config=configuration)
    if action == 1:
        play(observation.board, column=4, mark=1, config=configuration)
        action = agent(observation, configuration)
        assert action == 5
    elif action == 4:
        play(observation.board, column=1, mark=1, config=configuration)
        action = agent(observation, configuration)
        assert action == 0


@pytest.mark.parametrize("agent_name, agent", agents)
def test_single_column(observation, configuration, agent_name, agent):
    configuration.timeout = 60
    observation.mark  = 1
    observation.board = [
        2,0,2,2,1,0,2,
        1,0,1,1,2,0,1,
        2,0,2,2,1,0,2,
        1,0,1,1,2,0,2,
        2,0,2,2,1,0,1,
        1,0,1,2,2,1,2,
    ]
    action = agent(observation, configuration)
    assert action == 5
