
from kaggle_environments import make

from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent



def test_first_move_is_center():
    env = make("connectx", debug=True)
    env.configuration.timeout = 24*60*60
    action = AlphaBetaAgent.agent(env.state[0].observation, env.configuration, search_max_depth=3)
    assert action == 3  # always play the middle square first
    assert type(action) == int

def test_can_play_game_against_self():
    env = make("connectx", debug=True)
    env.run([AlphaBetaAgent.agent, AlphaBetaAgent.agent])
