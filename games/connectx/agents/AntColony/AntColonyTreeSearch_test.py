import pytest
from kaggle_environments import make

# from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AntColony.AntColonyTreeSearch import AntColonyTreeSearch


@pytest.fixture
def agent():
    return AntColonyTreeSearch

@pytest.fixture
def env():
    env = make("connectx", debug=True)
    env.configuration.timeout = 8
    return env

@pytest.fixture
def observation(env):
    return env.state[0].observation

@pytest.fixture
def configuration(env):
    return env.configuration

def test_first_move_is_center(agent, observation, configuration):
    action = agent(observation, configuration)
    assert action == 3   # always play the middle square first
    assert type(action) == int, f'should always play the center move first'


def test_can_play_game_against_self(agent):
    env = make("connectx", debug=True)
    env.configuration.timeout = 4
    env.run([ agent, agent ])

@pytest.mark.parametrize("position", ["player1","player2"])
@pytest.mark.parametrize("opponent", ["random", "negamax"])
def test_can_win_against_kaggle(agent, position, opponent):
    env = make("connectx", debug=True)
    env.configuration.timeout = 4  # Half normal timer for quicker tests
    if position == "player1":
        env.run([agent, opponent])
        assert env.state[0].reward == 1
        assert env.state[1].reward == -1
    else:
        env.run([opponent, agent])
        assert env.state[1].reward == 1
        assert env.state[0].reward == -1

#
# @pytest.mark.parametrize("position", ["player1","player2"])
# @pytest.mark.parametrize("opponent", ['AlphaBetaAgent'])
# def test_can_win_against_agents(agent, position, opponent):
#     if opponent == 'AlphaBetaAgent': opponent = AlphaBetaAgent.agent()
#     env = make("connectx", debug=True)
#     env.configuration.timeout = 8  # Half normal timer for quicker tests
#     if position == "player1":
#         env.run([agent, opponent])
#         assert env.state[0].reward == 1
#         assert env.state[1].reward == -1
#     else:
#         env.run([opponent, agent])
#         assert env.state[1].reward == 1
#         assert env.state[0].reward == -1
