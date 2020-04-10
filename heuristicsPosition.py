from typing import Tuple, Union


def manhattanDistance( position: Tuple[int], goal: Tuple[int], gameState=None ) -> int:
    "The Manhattan distance"
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def manhattanHeuristic( position: Tuple[int], problem: Union['SearchProblem', Tuple[int]] ) -> int:
    "The Manhattan distance heuristic for a PositionSearchProblem"
    goal = hasattr(problem, 'goal') and problem.goal or problem
    return manhattanDistance(position, goal)



def euclideanDistance( position: Tuple[int], goal: Tuple[int], gameState=None ) -> int:
    "The Euclidean distance"
    return ( (position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2 ) ** 0.5

def euclideanHeuristic( position: Tuple[int], problem: Union['SearchProblem', Tuple[int]] ) -> int:
    "The Euclidean distance heuristic for a PositionSearchProblem"
    goal = hasattr(problem, 'goal') and problem.goal or problem
    return euclideanDistance(position, goal)

