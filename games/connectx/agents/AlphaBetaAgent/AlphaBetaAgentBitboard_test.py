import pytest
from kaggle_environments import make

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaAgentBitboard import AlphaBetaAgentBitboard


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

@pytest.mark.parametrize("search_max_depth", range(1,8+1))
def test_first_move_is_center_depth(observation, configuration, search_max_depth):
    action = AlphaBetaAgentBitboard.agent(search_max_depth=search_max_depth)(observation, configuration)
    assert action == 3   # always play the middle square first
    assert type(action) == int, f'search_max_depth = {search_max_depth}'


def test_can_play_game_against_self():
    env = make("connectx", debug=True)
    env.configuration.timeout = 4
    env.run([ AlphaBetaAgentBitboard.agent(), AlphaBetaAgentBitboard.agent() ])

@pytest.mark.parametrize("position", ["player1","player2"])
@pytest.mark.parametrize("opponent", ["random", "negamax"])
def test_can_win_against_kaggle(position, opponent):
    env = make("connectx", debug=True)
    env.configuration.timeout = 4  # Half normal timer for quicker tests
    if position == "player1":
        env.run([AlphaBetaAgentBitboard.agent(), opponent])
        assert env.state[0].reward == 1
        assert env.state[1].reward == -1
    else:
        env.run([opponent, AlphaBetaAgentBitboard.agent()])
        assert env.state[1].reward == 1
        assert env.state[0].reward == -1


@pytest.mark.parametrize("position", ["player1","player2"])
@pytest.mark.parametrize("opponent", ['AlphaBetaAgent'])
def test_can_win_against_agents(position, opponent):
    if opponent == 'AlphaBetaAgent': opponent = AlphaBetaAgent.agent()
    env = make("connectx", debug=True)
    env.configuration.timeout = 8  # Half normal timer for quicker tests
    if position == "player1":
        env.run([AlphaBetaAgentBitboard.agent(), opponent])
        assert env.state[0].reward == 1
        assert env.state[1].reward == -1
    else:
        env.run([opponent, AlphaBetaAgentBitboard.agent()])
        assert env.state[1].reward == 1
        assert env.state[0].reward == -1
