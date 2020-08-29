import pytest

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.AlphaBetaAgent.AlphaBetaBitsquares import AlphaBetaBitsquares
from agents.AlphaBetaAgent.AlphaBetaOddEven import AlphaBetaOddEven
from agents.MontyCarlo.MontyCarloBitsquares import MontyCarloBitsquares
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloOddEven import MontyCarloOddEven
from agents.MontyCarlo.MontyCarloPure import MontyCarloPure
from agents.Negamax.Negamax import Negamax

agents = [
    # ('AntColonyTreeSearch',      AntColonyTreeSearch()),   # Unit tests currently broken
    ('MontyCarloBitsquares',     MontyCarloBitsquares()),
    ('MontyCarloOddEven',        MontyCarloOddEven()),
    ('MontyCarloHeuristic',      MontyCarloHeuristic()),
    ('MontyCarloPure',           MontyCarloPure()),
    ('Negamax',                  Negamax()),
    ('AlphaBetaAgent',           AlphaBetaAgent.agent()),
    ('AlphaBetaBitboard',        AlphaBetaBitboard.agent()),
    ('AlphaBetaBitboardEvenOdd', AlphaBetaOddEven.agent()),
    ('AlphaBetaBitsquares',      AlphaBetaBitsquares.agent()),
]

kaggle_agents = [
    "random",
    "negamax",
]


# Some of the agents have large datafiles, which might timeout on first move with a short test timer, so preload now
@pytest.mark.parametrize("agent_name, agent", agents)
def test_preload(observation, configuration, agent_name, agent):
    action = agent(observation, configuration)
    assert isinstance(action, int)