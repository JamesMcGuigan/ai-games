
from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristic
from agents.MontyCarlo.MontyCarloLinkedList import MontyCarloLinkedList
from agents.MontyCarlo.MontyCarloTreeSearch import MontyCarloTreeSearch
from agents.Negamax.Negamax import Negamax



agents = [
    ('MontyCarloHeuristic',  MontyCarloHeuristic()),
    ('MontyCarloLinkedList', MontyCarloLinkedList()),
    ('MontyCarloTreeSearch', MontyCarloTreeSearch),
    ('Negamax',              Negamax()),
    ('AlphaBetaAgent',       AlphaBetaAgent.agent()),
    ('AlphaBetaBitboard',    AlphaBetaBitboard.agent()),
]

kaggle_agents = [
    "random",
    "negamax",
]
