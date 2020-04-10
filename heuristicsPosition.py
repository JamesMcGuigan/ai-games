from typing import Tuple


def manhattanDistance( position: Tuple[int], goal: Tuple[int] ) -> int:
    "The Manhattan distance"
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def manhattanHeuristic( position, problemOrGoal ):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = hasattr(problemOrGoal, 'goal') and problemOrGoal.goal or problemOrGoal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanDistance( position: Tuple[int], goal: Tuple[int] ) -> int:
    "The Euclidean distance"
    return ( (position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2 ) ** 0.5

def euclideanHeuristic( position, problemOrGoal ):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = hasattr(problemOrGoal, 'goal') and problemOrGoal.goal or problemOrGoal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
