### Results:
### Sum
### - produces much better results than: Max, Mean, Median, Min, Null
### - on tinyCorners:                 EuclideanSum produces better path and fewer expanded nodes than ManhattenSum
### - on mediumCorners + bigCorners:  equal path length - but ManhattenSum expands much fewer nodes
### CornersPath
### - produces comparable or even better results than ManhattenSum
### - min() is worst
### - max() is sometimes good (on mediumCorners)
### - mean() + median() produces most consistent good results - use median()
### - this results in tradeoff between heading straight to the nearest corner, a more diffuse search exploring all corners

### HEURS=`cat searchAgents.py | grep '^def cornersHeuristic' | awk -F'[ (]' '{ print $2 }'`
### LAYOUTS="tinyCorners mediumCorners bigCorners"
### (for LAYOUT in $LAYOUTS; do for HEUR in $HEURS; do (python ./pacman.py -p SearchAgent -a fn=astar,prob=CornersProblem,heuristic=$HEUR -q -l $LAYOUT | grep '\bPath\|Scores:\|nodes') | perl -p -e 's/\s+/ /; s/\n/ - /'; echo -e "$LAYOUT - $HEUR"; done; done;) | sort -n -k15

# Path found with total cost of  30 in 0.0 seconds - Search nodes expanded:  159 - Scores: 510.0 - tinyCorners - cornersHeuristicCornersPathMean
# Path found with total cost of  30 in 0.0 seconds - Search nodes expanded:  162 - Scores: 510.0 - tinyCorners - cornersHeuristicCornersPathMedian
# Path found with total cost of  32 in 0.0 seconds - Search nodes expanded:  162 - Scores: 508.0 - tinyCorners - cornersHeuristicCornersPathMax
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  163 - Scores: 512.0 - tinyCorners - cornersHeuristicCornersPath
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  163 - Scores: 512.0 - tinyCorners - cornersHeuristicCornersPathMin
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  178 - Scores: 512.0 - tinyCorners - cornersHeuristicEuclideanSum
# Path found with total cost of  32 in 0.0 seconds - Search nodes expanded:  193 - Scores: 508.0 - tinyCorners - cornersHeuristic
# Path found with total cost of  32 in 0.0 seconds - Search nodes expanded:  193 - Scores: 508.0 - tinyCorners - cornersHeuristicManhattenSum
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  199 - Scores: 512.0 - tinyCorners - cornersHeuristicManhattenMax
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  205 - Scores: 512.0 - tinyCorners - cornersHeuristicEuclideanMax
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  205 - Scores: 512.0 - tinyCorners - cornersHeuristicManhattenMean
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  208 - Scores: 512.0 - tinyCorners - cornersHeuristicManhattenMedian
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  213 - Scores: 512.0 - tinyCorners - cornersHeuristicEuclideanMean
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  213 - Scores: 512.0 - tinyCorners - cornersHeuristicEuclideanMedian
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  231 - Scores: 512.0 - tinyCorners - cornersHeuristicManhattenMin
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  233 - Scores: 512.0 - tinyCorners - cornersHeuristicEuclideanMin
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  248 - Scores: 512.0 - tinyCorners - cornersHeuristicNull

# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  502 - Scores: 434.0 - mediumCorners - cornersHeuristic
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  502 - Scores: 434.0 - mediumCorners - cornersHeuristicManhattenSum
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  548 - Scores: 434.0 - mediumCorners - cornersHeuristicCornersPathMax
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  573 - Scores: 434.0 - mediumCorners - cornersHeuristicCornersPathMedian
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  577 - Scores: 434.0 - mediumCorners - cornersHeuristicCornersPathMean
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  703 - Scores: 434.0 - mediumCorners - cornersHeuristicEuclideanSum
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  774 - Scores: 434.0 - mediumCorners - cornersHeuristicCornersPath
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded:  774 - Scores: 434.0 - mediumCorners - cornersHeuristicCornersPathMin
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1136 - Scores: 434.0 - mediumCorners - cornersHeuristicManhattenMax
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1241 - Scores: 434.0 - mediumCorners - cornersHeuristicEuclideanMax
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1289 - Scores: 434.0 - mediumCorners - cornersHeuristicManhattenMean
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1336 - Scores: 434.0 - mediumCorners - cornersHeuristicManhattenMedian
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1390 - Scores: 434.0 - mediumCorners - cornersHeuristicEuclideanMean
# Path found with total cost of 106 in 0.2 seconds - Search nodes expanded: 1421 - Scores: 434.0 - mediumCorners - cornersHeuristicEuclideanMedian
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1475 - Scores: 434.0 - mediumCorners - cornersHeuristicManhattenMin
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1532 - Scores: 434.0 - mediumCorners - cornersHeuristicEuclideanMin
# Path found with total cost of 106 in 0.0 seconds - Search nodes expanded: 1960 - Scores: 434.0 - mediumCorners - cornersHeuristicNull

# Path found with total cost of 162 in 0.1 seconds - Search nodes expanded: 1028 - Scores: 378.0 - bigCorners - cornersHeuristicCornersPathMedian
# Path found with total cost of 162 in 0.1 seconds - Search nodes expanded: 1105 - Scores: 378.0 - bigCorners - cornersHeuristicCornersPathMean
# Path found with total cost of 162 in 0.0 seconds - Search nodes expanded: 1207 - Scores: 378.0 - bigCorners - cornersHeuristic
# Path found with total cost of 162 in 0.0 seconds - Search nodes expanded: 1207 - Scores: 378.0 - bigCorners - cornersHeuristicManhattenSum
# Path found with total cost of 162 in 0.1 seconds - Search nodes expanded: 1726 - Scores: 378.0 - bigCorners - cornersHeuristicCornersPath
# Path found with total cost of 162 in 0.1 seconds - Search nodes expanded: 1726 - Scores: 378.0 - bigCorners - cornersHeuristicCornersPathMin
# Path found with total cost of 162 in 0.1 seconds - Search nodes expanded: 1883 - Scores: 378.0 - bigCorners - cornersHeuristicCornersPathMax
# Path found with total cost of 166 in 0.1 seconds - Search nodes expanded: 2509 - Scores: 374.0 - bigCorners - cornersHeuristicEuclideanSum
# Path found with total cost of 162 in 0.2 seconds - Search nodes expanded: 4380 - Scores: 378.0 - bigCorners - cornersHeuristicManhattenMax
# Path found with total cost of 162 in 0.6 seconds - Search nodes expanded: 5182 - Scores: 378.0 - bigCorners - cornersHeuristicManhattenMedian
# Path found with total cost of 162 in 0.3 seconds - Search nodes expanded: 5191 - Scores: 378.0 - bigCorners - cornersHeuristicManhattenMean
# Path found with total cost of 162 in 0.3 seconds - Search nodes expanded: 5196 - Scores: 378.0 - bigCorners - cornersHeuristicEuclideanMax
# Path found with total cost of 162 in 0.7 seconds - Search nodes expanded: 5540 - Scores: 378.0 - bigCorners - cornersHeuristicEuclideanMedian
# Path found with total cost of 162 in 0.4 seconds - Search nodes expanded: 5559 - Scores: 378.0 - bigCorners - cornersHeuristicEuclideanMean
# Path found with total cost of 162 in 0.3 seconds - Search nodes expanded: 5850 - Scores: 378.0 - bigCorners - cornersHeuristicManhattenMin
# Path found with total cost of 162 in 0.4 seconds - Search nodes expanded: 6073 - Scores: 378.0 - bigCorners - cornersHeuristicEuclideanMin
# Path found with total cost of 162 in 0.2 seconds - Search nodes expanded: 7907 - Scores: 378.0 - bigCorners - cornersHeuristicNull
from functools import lru_cache
from itertools import permutations
from typing import Tuple, Union

import numpy as np

from heuristicsPosition import euclideanDistance, manhattanDistance



def cornerDistances(state, type='manhatten'):
    (position, corners) = state
    if type == 'manhatten': return [ manhattanDistance(position, corner) for corner in corners ]
    if type == 'euclidean': return [ euclideanDistance(position, corner) for corner in corners ]
    raise Exception('cornerDistances() - undefined type: ' + str(type))


# NOTE: This is only optimized for short lists
@lru_cache(None)
def shortestPathBetweenCorners(corners: Tuple[tuple]) -> Union[float,int]:
    if len(corners) == 2: return manhattanDistance(corners[0], corners[1])
    if len(corners) == 1: return 0
    if len(corners) == 0: return 0

    path_costs = {}
    for path in permutations(corners):
        if reversed(path) in path_costs: continue  # reversed cost is same as forward cost
        path_costs[path] = sum([
            manhattanDistance(path[n], path[n+1])
            for n in range(0, len(path)-1)
        ])
    min_cost = min(path_costs.values())
    return min_cost


# for PROB in tinyCorners mediumCorners bigCorners; do (python ./pacman.py -p SearchAgent -a fn=astar,prob=CornersProblem,heuristic=cornersHeuristicNull -q -l $PROB | grep 'Path\|Scores:\|nodes') | perl -p -e 's/\s+/ /; s/\n/ - /'; echo -e "$PROB"; done;
# Path found with total cost of  28 in 0.0 seconds - Search nodes expanded:  254 - Scores: 512.0 - tinyCorners
# Path found with total cost of 106 in 0.1 seconds - Search nodes expanded: 1954 - Scores: 434.0 - mediumCorners
# Path found with total cost of 162 in 0.3 seconds - Search nodes expanded: 7946 - Scores: 378.0 - bigCorners
def cornersHeuristicNull(state, problem):
    return 0

def cornersHeuristicManhattenMin(state, problem):
    distances = cornerDistances(state, type='manhatten')
    return len(distances) and np.min(distances) or 0

def cornersHeuristicManhattenMean(state, problem):
    distances = cornerDistances(state, type='manhatten')
    return len(distances) and np.mean(distances) or 0

def cornersHeuristicManhattenMedian(state, problem):
    distances = cornerDistances(state, type='manhatten')
    return len(distances) and np.median(distances) or 0

def cornersHeuristicManhattenMax(state, problem):
    distances = cornerDistances(state, type='manhatten')
    return len(distances) and np.max(distances) or 0

# Sum is a good, but inadmissible heuristic
def cornersHeuristicManhattenSum(state, problem):
    distances = cornerDistances(state, type='manhatten')
    return len(distances) and sum(distances) or 0


def cornersHeuristicEuclideanMin(state, problem):
    distances = cornerDistances(state, type='euclidean')
    return len(distances) and np.min(distances) or 0

def cornersHeuristicEuclideanMean(state, problem):
    distances = cornerDistances(state, type='euclidean')
    return len(distances) and np.mean(distances) or 0

def cornersHeuristicEuclideanMedian(state, problem):
    distances = cornerDistances(state, type='euclidean')
    return len(distances) and np.median(distances) or 0

def cornersHeuristicEuclideanMax(state, problem):
    distances = cornerDistances(state, type='euclidean')
    return len(distances) and np.max(distances) or 0

def cornersHeuristicEuclideanSum(state, problem):
    distances = cornerDistances(state, type='euclidean')
    return len(distances) and sum(distances) or 0


# for PROB in tinyCorners mediumCorners bigCorners; do (python ./pacman.py -p SearchAgent -a fn=astar,prob=CornersProblem,heuristic=cornersHeuristicMinCornersPath -q -l $PROB | grep 'Path\|Scores:\|nodes') | perl -p -e 's/\s+/ /; s/\n/ - /'; echo -e "$PROB"; done;
def cornersHeuristicCornersPath(state, problem, function=np.min):
    (position, corners) = state
    if len(corners) == 0: return 0

    cost = function([
        manhattanDistance(position, corner) + shortestPathBetweenCorners(corners)
        for corner in corners
        ])
    return cost

def cornersHeuristicCornersPathMin(state, problem):
    return cornersHeuristicCornersPath(state, problem, function=np.min)

def cornersHeuristicCornersPathMean(state, problem):
    return cornersHeuristicCornersPath(state, problem, function=np.mean)

def cornersHeuristicCornersPathMedian(state, problem):
    return cornersHeuristicCornersPath(state, problem, function=np.median)

def cornersHeuristicCornersPathMax(state, problem):
    return cornersHeuristicCornersPath(state, problem, function=np.max)



# python autograder.py -q q6
# find layouts -name '*Corners*' | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=bfs,prob=CornersProblem,heuristic=cornersHeuristic -q -l | grep nodes
def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    (position, corners) = state
    corners = problem.corners # These are the corner coordinates
    walls   = problem.walls # These are the walls of the maze, as a Grid (game.py)

    ### Admissibility vs. Consistency:
    ### To be admissible, the heuristic values must be lower bounds on the actual shortest path cost to the nearest goal (and non-negative).
    ### To be consistent, it must additionally hold that if an action has cost c, then taking that action can only cause a drop in heuristic of at most c.
    ### Inadmissible heuristics overestimate maximum distance
    ### inconsistent heuristics do not monotomically decrease towards the goal

    return cornersHeuristicCornersPathMin(state, problem)        # PASS: Heuristic resulted in expansion of 774 nodes
    # return cornersHeuristicCornersPathMax(state, problem)      # FAIL: Inadmissible + inconsistent | 548 nodes
    # return cornersHeuristicCornersPathMedian(state, problem)   # FAIL: inconsistent heuristic | 502 nodes
    # return cornersHeuristicCornersPathMean(state, problem)     # FAIL: inconsistent heuristic | 577 nodes

    # return cornersHeuristicManhattenMax(state, problem)        # PASS: Heuristic resulted in expansion of 1136 nodes
    # return cornersHeuristicEuclideanSum(state, problem)        # FAIL: inconsistent heuristic | 703 nodes
    # return cornersHeuristicManhattenSum(state, problem)        # FAIL: Inadmissible heuristic | 502 nodes
