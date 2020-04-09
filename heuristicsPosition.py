def manhattanHeuristic( position, problemOrGoal, info={} ):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = hasattr(problemOrGoal, 'goal') and problemOrGoal.goal or problemOrGoal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic( position, problemOrGoal, info={} ):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = hasattr(problemOrGoal, 'goal') and problemOrGoal.goal or problemOrGoal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
