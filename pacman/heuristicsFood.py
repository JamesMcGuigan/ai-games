from functools import lru_cache
from typing import FrozenSet, List, Tuple, Union

from sortedcollections import ValueSortedDict

from heuristicsPosition import manhattanDistance
from search import astar, mazeDistance



########################################################################################################################

_current_problem_hash = None
def resetStateOnNewProblem( problem ):
    global _current_problem_hash
    global _optimalMazePath
    if _current_problem_hash != hash(problem):
        _current_problem_hash = hash(problem)
        _distanceCache        = {}
        _optimalMazePath      = {}

########################################################################################################################

_distanceCache  = {}  # { goal: ValueSortedDict({ peer: cost }) }
def updateDistanceCache(goals: Union[FrozenSet,Tuple], distance=manhattanDistance):
    global _distanceCache
    if len(set(goals) - set(_distanceCache.keys())) == 0: return

    for goal in goals:
        if goal in _distanceCache: continue
        _distanceCache[goal] = ValueSortedDict()
        for peer in (set(goals) | set(_distanceCache.keys())):
            if peer == goal:                 continue   # distance to self is zero
            if peer in _distanceCache[goal]: continue   # lookup exists
            if peer not in _distanceCache:   _distanceCache[peer] = ValueSortedDict()

            twin_lookup = _distanceCache.get(peer,{}).get(goal)
            if twin_lookup:
                _distanceCache[goal][peer] = twin_lookup
            else:
                _distanceCache[goal][peer] = _distanceCache[peer][goal] = distance(goal, peer)


def distanceCacheDistance(source: Tuple[int,int], goal: Tuple[int,int], distance=manhattanDistance):
    global _distanceCache
    return _distanceCache.get(source,{}).get(goal,None) or distance(source, goal)


@lru_cache(2**16)
def getClosestGoalCost(source: Tuple[int,int], goals: Union[FrozenSet,Tuple], distance=manhattanDistance) -> (int, Tuple[int,int]):
    if len(goals) == 0: return (0, source)
    if len(goals) == 1: return (distance(source, list(goals)[0]), list(goals)[0])

    goals = frozenset(goals) - {source}
    updateDistanceCache(goals, distance=distance)
    if source in _distanceCache:
        for target, cost in _distanceCache[source].items():
            if target in goals:
                return (cost, target)

    # We may not be searching from a food square, so compute manually
    # QUESTION: Should this be stored in _distanceCache - No - @lru_cache is sufficent
    return min([ (distance(source, goal), goal) for goal in goals ])


@lru_cache(2**16)
def getNeighbours(source: Tuple[int,int], goals: FrozenSet[Tuple[int,int]]) -> FrozenSet[Tuple[int,int]]:
    x = source[0]
    y = source[0]
    neighbours = frozenset([
        (x1,y1) for x1 in [x-1,x,x+1] for y1 in [y-1,y,y+1]
        if (x1,y1) != (x,y) and (x1,y1) in goals
    ])
    return neighbours


@lru_cache(2**16)
def getClosestGoalPath(source: Tuple[int,int], goals: Union[FrozenSet,Tuple], distance=manhattanDistance) -> (int, List[Tuple[int,int]]):
    goals = frozenset(goals) - {source}

    if len(goals) == 0: return (0, [])
    if len(goals) == 1: return (distance(source, list(goals)[0]), list(goals))

    # NOTE: Adding a loop here over the several next closest does not improve the score
    # Optimization: recursion + lru_cache = fast
    edge_cost, next_goal = getClosestGoalCost(source,    goals,               distance=distance)
    path_cost, full_path = getClosestGoalPath(next_goal, goals - {next_goal}, distance=distance)
    cost = edge_cost   + path_cost
    path = [next_goal] + full_path
    return (cost, path)


@lru_cache(2**16)
def getBestGoalPath(source: Tuple[int,int], goals: Union[FrozenSet,Tuple], distance=manhattanDistance) -> (int, List[Tuple[int,int]]):
    """
    Find the absolute best path, by comparing all possible combinations of paths
    """
    goals = frozenset(goals) - {source}

    if len(goals) == 0: return [0, []]
    if len(goals) == 1: return [distance(source, list(goals)[0]), []]

    paths = []
    shortlist = getNeighbours(source, goals) or goals
    for next_goal in shortlist:
        edge_cost = distance(source, next_goal)
        path_cost, full_path = getBestGoalPath(next_goal, shortlist - {source,next_goal}, distance=distance)
        cost  = edge_cost + path_cost
        path  = [next_goal] + full_path
        paths.append((cost,path))
    return min(paths)


@lru_cache(2**16)
def getPathCosts(paths: Tuple[Tuple[int,int]], distance=manhattanDistance):
    if len(paths) == 0: return [0]
    if len(paths) == 1: return [0]

    costs = [ distance(paths[n], paths[n+1]) for n in range(len(paths)-1) ]
    return costs

@lru_cache(2**16)
def getPathCost(paths: Tuple[Tuple[int,int]], distance=manhattanDistance):
    if len(paths) == 0: return 0
    if len(paths) == 1: return 0
    return sum(getPathCosts(paths))






def foodHeuristicClosestGoal( state, problem, distance=manhattanDistance) -> int:
    """This simply returns the manhattanDistance to the closest food item"""
    position, foodGrid = state
    cost, goal = getClosestGoalCost(position, frozenset(foodGrid.asList()), distance=distance)
    return cost

def foodHeuristicClosestPath(state, problem, distance=manhattanDistance) -> int:
    """This simply returns the manhattanDistance to the closest food item"""
    resetStateOnNewProblem(problem)

    position, foodGrid = state
    goals = frozenset(foodGrid.asList())

    if len(goals) == 0: return 0
    if len(goals) == 1: return distance(position, list(goals)[0])

    # Search each of the possible goals, and determine which one produces the best total path
    costs = []
    for goal in goals:
        edge_cost, next_goal = getClosestGoalCost(position,  goals - {position},            distance=distance)
        path_cost, path      = getClosestGoalPath(next_goal, goals - {position, next_goal}, distance=distance)
        costs.append(edge_cost + path_cost)
    return min(costs) if len(costs) else 0


########################################################################################################################


# 1/4 - Unexpectedly bad - this is worse than just len(foodGrid.asList()) - expanded nodes: 17268
_optimalMazePath = {} # List[Tuple[int,int]] = [paths]
def foodHeuristicFollowMazePath(state, problem, best=False, recompute_after_goal=False, add_path_cost=True) -> int:
    resetStateOnNewProblem(problem)

    global _optimalMazePath
    position, foodGrid = state
    goals              = frozenset(foodGrid.asList())

    if len(goals) == 0: return 0
    if len(goals) == 1: return manhattanDistance(position, list(goals)[0])

    # On first iteration compute optimalMazePath
    key         = goals if recompute_after_goal else True  # True = compute once | goals = recompute after each goal
    visualize   = False
    algorithm   = astar
    # algorithm   = gdfs

    distance    = lambda p1, p2: mazeDistance(p1, p2, gameState=problem.startingGameState, algorithm=algorithm, visualize=visualize)
    if key not in _optimalMazePath:
        if best:
            _optimalMazePath[key] = getBestGoalPath(position, goals-{position}, distance=distance)  # This is very slow
        else:
            # Consider each possible goal, and compute a closest path route cost
            paths = []
            for goal in goals:
                edge_cost       = distance(position, goal)
                path_cost, path = getClosestGoalPath(goal, goals - {goal,position}, distance=distance)
                cost            = edge_cost + path_cost
                paths.append( (cost, path, path_cost) )
            _optimalMazePath[key] = min(paths)
        # print('_optimalMazePath', _optimalMazePath[key])

    # Then we just follow the path, using manhatten distance to the next goal in sequence
    for next_goal in _optimalMazePath[key][1]:
        if next_goal not in goals: continue

        remaining_goals = tuple([ goal for goal in _optimalMazePath[key][1] if goal in goals and goals != next_goal ])
        edge_cost = mazeDistance(position, next_goal,
                gameState=problem.startingGameState,
                algorithm=algorithm,
                visualize=visualize
        )
        if best or add_path_cost:
            path_cost = getPathCost(remaining_goals, distance=distance)
        else:
            # +path_cost makes everything quick, but fails admissibility without perfect pathing
            path_cost = 0

        return edge_cost + path_cost
        # return manhattanDistance(position, next_goal)   # trickySearch =  60 in 4.1 seconds + 8064 nodes - 3/4  expanded nodes: 8064
    return 0

########################################################################################################################


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


    # Boards: bigSafeSearch, bigSearch, boxSearch, greedySearch, mediumSafeSearch, mediumSearch, oddSearch, openSearch
    #         smallSafeSearch, smallSearch, testSearch, tinySafeSearch, tinySearch, trickySearch
    # python autograder.py -q q7
    # python pacman.py -p AStarFoodSearchAgent -l testSearch
    # python pacman.py -p AStarFoodSearchAgent -l trickySearch
    # python pacman.py -p AStarFoodSearchAgent -l greedySearch
    # find ./layouts -name '*Search*' | grep -v 'big\|mediumSearch\|boxSearch' | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p AStarFoodSearchAgent -l


    resetStateOnNewProblem(problem)
    position, foodGrid = state

    return foodHeuristicFollowMazePath(state, problem, best=True, recompute_after_goal=True)     ### 5/4 - expanded nodes:  6677
    # return foodHeuristicFollowMazePath(state, problem, best=True, recompute_after_goal=False)  ### 2/4 - expanded nodes: 10028
    # return foodHeuristicFollowMazePath(state, problem, best=False, recompute_after_goal=True,  add_path_cost=False)   ### 4/4 - expanded nodes:  7891
    # return foodHeuristicFollowMazePath(state, problem, best=False, recompute_after_goal=False, add_path_cost=False)   ### 4/4 - expanded nodes:  8068

    # return foodHeuristicFollowMazePath(state, problem, best=False, recompute_after_goal=True,  add_path_cost=True)  # fast but fails admissibility


    ### 3/4 - expanded nodes:  9625  | fails admissibility without * 0.5
    # return foodHeuristicClosestPath(state, problem, distance=euclideanDistance) * 0.5

    ### 3/4 - expanded nodes:  9093  | fails admissibility without * 0.5
    # return foodHeuristicClosestPath(state, problem, distance=manhattanDistance) * 0.5

    # return foodHeuristicClosestGoal(state, problem)      # 2/4 - expanded nodes: 13898
    # return len(foodGrid.asList())                        # 2/4 - expanded nodes: 12517

    ### Works well, but fails both admissibility and consistency tests
    # return foodHeuristicClosestPath(state, problem, distance=lambda p1, p2: mazeDistance(p1, p2, problem.startingGameState))

