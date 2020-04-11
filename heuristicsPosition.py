from functools import lru_cache
from typing import Tuple, Union

import search
from searchProblems import PositionSearchProblem



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



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


@lru_cache(2**16)
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    try:
        x1, y1 = point1
        x2, y2 = point2
        walls = gameState.getWalls()
        assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
        assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
        prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
        return len(search.bfs(prob))
    except:
        return 0
