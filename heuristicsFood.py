from functools import lru_cache
from typing import FrozenSet, List, Tuple, Union

from sortedcollections import ValueSortedDict

from heuristicsPosition import manhattanDistance



_distanceCache  = {}  # { goal: ValueSortedDict({ peer: cost }) }
_pathCostsCache = {}

def updateDistanceCache(goals: Union[FrozenSet,Tuple]):
    if len(set(goals) - set(_distanceCache.keys())) == 0: return

    for goal in goals:
        if goal in _distanceCache: continue
        _distanceCache[goal] = ValueSortedDict()
        for peer in (set(goals) | set(_distanceCache.keys())):
            if peer in _distanceCache[goal]: continue   # lookup exists
            if peer not in _distanceCache:   _distanceCache[peer] = ValueSortedDict()

            twin_lookup = _distanceCache.get(peer,{}).get(goal)
            if twin_lookup:
                _distanceCache[goal][peer] = twin_lookup
            else:
                _distanceCache[goal][peer] = _distanceCache[peer][goal] = manhattanDistance(goal, peer)


@lru_cache(2**16)
def getClosestGoalCost(source: Tuple[int], goals: Union[FrozenSet,Tuple]) -> (int, Tuple[int]):
    if len(goals) == 0: return (0, source)
    if len(goals) == 1: return (manhattanDistance(source, list(goals)[0]), list(goals)[0])

    goals = frozenset(goals) - {source}
    updateDistanceCache(goals)
    if source in _distanceCache:
        for target, cost in _distanceCache[source].items():
            if target in goals:
                return (cost, target)

    # We may not be searching from a food square, so compute manually
    # QUESTION: Should this be stored in _distanceCache - No - @lru_cache is sufficent
    return min([ (manhattanDistance(source, goal), goal) for goal in goals ])


@lru_cache(2**16)
def getClosestGoalPath(source: Tuple[int], goals: Union[FrozenSet,Tuple]) -> (int, List[Tuple[int]]):
    goals = frozenset(goals) - {source}

    if len(goals) == 0: return (0, [])
    if len(goals) == 1: return (manhattanDistance(source, list(goals)[0]), list(goals))

    edge_cost, next_goal = getClosestGoalCost(source,    goals)
    path_cost, full_path = getClosestGoalPath(next_goal, goals - {next_goal})  # recursion + lru_cache = fast

    cost = edge_cost   + path_cost
    path = [next_goal] + full_path
    return (cost, path)



def foodHeuristicClosestGoal( state, problem ) -> int:
    """This simply returns the manhattanDistance to the closest food item"""
    position, foodGrid = state
    cost, goal = getClosestGoalCost(position, frozenset(foodGrid.asList()))
    return cost

def foodHeuristicClosestPath(state, problem) -> int:
    """This simply returns the manhattanDistance to the closest food item"""
    position, foodGrid = state
    goals = frozenset(foodGrid.asList())

    if len(goals) == 0: return 0
    if len(goals) == 1: return manhattanDistance(position, list(goals)[0])

    costs = []
    for goal in goals:
        edge_cost, next_goal = getClosestGoalCost(position,  goals - {position})
        path_cost, path      = getClosestGoalPath(next_goal, goals - {position, next_goal})
        costs.append(edge_cost + path_cost)
    return min(costs) if len(costs) else 0





# Boards: bigSafeSearch, bigSearch, boxSearch, greedySearch, mediumSafeSearch, mediumSearch, oddSearch, openSearch
#         smallSafeSearch, smallSearch, testSearch, tinySafeSearch, tinySearch, trickySearch
# python autograder.py -q q7
# python pacman.py -p AStarFoodSearchAgent -l testSearch
# python pacman.py -p AStarFoodSearchAgent -l trickySearch
# python pacman.py -p AStarFoodSearchAgent -l greedySearch
# find ./layouts -name '*Search*' | grep -v 'big\|mediumSearch' | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p AStarFoodSearchAgent -l

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state

    # python autograder.py -q q7
    return foodHeuristicClosestPath(state, problem) * 0.5  # 3/4 - expanded nodes:  9093  | fails admissibility without * 0.5
    # return foodHeuristicClosestGoal(state, problem)      # 2/4 - expanded nodes: 13898
    # return len(foodGrid.asList())                        # 2/4 - expanded nodes: 12517

