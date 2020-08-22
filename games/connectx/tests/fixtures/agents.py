
from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.AlphaBetaAgent.AlphaBetaBitboardEvenOdd import AlphaBetaBitboardEvenOdd
from agents.AlphaBetaAgent.AlphaBetaBitsquares import AlphaBetaBitsquares
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloPure import MontyCarloPure
from agents.Negamax.Negamax import Negamax

agents = [
    # ('AntColonyTreeSearch',      AntColonyTreeSearch()),   # Unit tests currently broken
    ('MontyCarloHeuristic',      MontyCarloHeuristic()),
    ('MontyCarloPure',           MontyCarloPure()),
    ('Negamax',                  Negamax()),
    ('AlphaBetaAgent',           AlphaBetaAgent.agent()),
    ('AlphaBetaBitboard',        AlphaBetaBitboard.agent()),
    ('AlphaBetaBitboardEvenOdd', AlphaBetaBitboardEvenOdd.agent()),
    ('AlphaBetaBitsquares',      AlphaBetaBitsquares.agent()),
]

kaggle_agents = [
    "random",
    "negamax",
]
