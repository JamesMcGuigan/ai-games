
from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.AntColonyTreeSearch import AntColonyTreeSearch
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloPure import MontyCarloPure
from agents.Negamax.Negamax import Negamax



agents = [
    ('AntColonyTreeSearch',  AntColonyTreeSearch()),
    ('MontyCarloHeuristic',  MontyCarloHeuristic()),
    ('MontyCarloPure',       MontyCarloPure()),
    ('Negamax',              Negamax()),
    ('AlphaBetaAgent',       AlphaBetaAgent.agent()),
    ('AlphaBetaBitboard',    AlphaBetaBitboard.agent()),
]

kaggle_agents = [
    "random",
    "negamax",
]
