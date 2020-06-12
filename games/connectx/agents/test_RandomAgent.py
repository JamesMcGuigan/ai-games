
from kaggle_environments import make

from games.connectx.agents.RandomAgent import RandomAgent



def test_first_move():
    env = make("connectx", debug=True)
    env.configuration.timeout = 24*60*60
    action = RandomAgent.agent(env.state[0].observation, env.configuration, search_max_depth=3)
    assert 0 <= action < 7
    assert type(action) == int

def test_can_play_game_against_self():
    env = make("connectx", debug=False)
    env.run([RandomAgent.agent, RandomAgent.agent])
