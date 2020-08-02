import pytest

from tests.fixtures.agents import agents
from tests.fixtures.agents import kaggle_agents



@pytest.mark.parametrize("position", ["player1","player2"])
@pytest.mark.parametrize("agent_name, agent", agents)
@pytest.mark.parametrize("opponent", kaggle_agents)
def test_can_win_against_kaggle(env, position, agent_name, agent, opponent):
    env.configuration.timeout = 4  # Half normal timer for quicker tests
    if position == "player1":
        env.run([agent, opponent])
        assert env.state[0].reward == 1
        assert env.state[1].reward == -1
    else:
        env.run([opponent, agent])
        assert env.state[1].reward == 1
        assert env.state[0].reward == -1
