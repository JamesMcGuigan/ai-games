# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import random
from typing import Union, List, Dict, Tuple, Callable

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def distanceHeuristic(state, problem: SearchProblem=None):
    """
    Compute the manhattanDistance() to the goal, if defined
    NOTE: problem.goal is not defined in unit tests
    """
    if problem and hasattr(problem, 'goal'):
        return util.manhattanDistance(state, problem.goal)
    else:
        return 0


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# python autograder.py -q q1
# python pacman.py -l tinyMaze   -p SearchAgent -a fn=dfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=dfs
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=dfs -l
# Mazes: bigMaze contoursMaze mediumDottedMaze mediumMaze mediumScaryMaze openMaze smallMaze testMaze tinyMaze
def depthFirstSearch(
        problem:   SearchProblem,
        state:     Union[str,Tuple[int]] = None,  # Tuple[int] for runtime, str for unit tests
        actions:   List[str]             = None,
        visited:   Dict[Tuple[int],bool] = None,
        heuristic: Callable[ [Union[str,Tuple[int]], SearchProblem], int] = None
) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    state   = state   or problem.getStartState()
    actions = actions or []
    visited = visited or {}
    visited[state] = True

    if problem.isGoalState(state):
        return actions
    else:
        successors = problem.getSuccessors(state)

        ### Greedy Depth First Search
        def sort_fn(successor):
            # Will sort by lowest cost first, then add the cost of the (optional) heuristic
            # NOTE: problem.goal is not defined in unit tests
            (state, action, cost) = successor
            if callable(heuristic):
                cost += heuristic(state, problem)
            return cost
        successors = sorted(successors, key=sort_fn)

        ### Recursively traverse search tree
        for (next_state, next_action, next_cost) in successors:
            if next_state in visited: continue    # avoid searching already explored states
            ### add the next action to the list, and see if this path finds the goal, else backtrack
            next_actions = actions + [next_action]
            next_actions = depthFirstSearch(
                problem   = problem,
                state     = next_state,
                actions   = next_actions,
                visited   = visited,
                heuristic = heuristic,
            )
            if not next_actions:       # have we have hit a dead end
                continue               # try the next available path
            else:
                return next_actions    # return and save results

        ### if len(successors) == 0 or all successors returned false
        return []                      # backtrack



# python pacman.py -l tinyMaze   -p SearchAgent -a fn=gdfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=gdfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=gdfs
# find layouts -name '*Corners*' | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=gdfs -l
# Mazes: bigMaze contoursMaze mediumDottedMaze mediumMaze mediumScaryMaze openMaze smallMaze testMaze tinyMaze
def greedyDepthFirstSearch(
        problem: SearchProblem,
        state:   Tuple[int]            = None,
        actions: List[str]             = None,
        visited: Dict[Tuple[int],bool] = None,
) -> Union[List[str], bool]:
    return depthFirstSearch(
        problem   = problem,
        state     = state,
        actions   = actions,
        visited   = visited,
        heuristic = distanceHeuristic
    )


# python autograder.py -q q2
# python pacman.py -l tinyMaze   -p SearchAgent -a fn=bfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=bfs
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=bfs -l
# Mazes: bigMaze contoursMaze mediumDottedMaze mediumMaze mediumScaryMaze openMaze smallMaze testMaze tinyMaze
def breadthFirstSearch(
        problem:      SearchProblem,
        heuristic:    Callable[ [Union[str,Tuple[int]], SearchProblem], int] = None,
) -> List[str]:
    """
    Search the shallowest nodes in the search tree first.

    # https://en.wikipedia.org/wiki/Breadth-first_search
    procedure BFS(G, start_v) is
        let Q be a queue
        label start_v as discovered
        Q.enqueue(start_v)
        while Q is not empty do
            v := Q.dequeue()
            if v is the goal then
                return v
            for all edges from v to w in G.adjacentEdges(v) do
                if w is not labeled as discovered then
                    label w as discovered
                    w.parent := v
                    Q.enqueue(w)
    """
    state    = problem.getStartState()
    frontier = util.PriorityQueue()
    visited  = {}
    visited[state] = True

    frontier.push((state, []), 0)  # initialize root node, and add as first item in the queue

    while not frontier.isEmpty():
        (state, action_path) = frontier.pop()    # inspect next shortest path

        if problem.isGoalState(state):
            return action_path

        for (child_state, child_action, child_cost) in problem.getSuccessors(state):
            if child_state in visited: continue  # avoid searching already explored states

            visited[child_state] = True                                           # mark as seen
            child_path = action_path + [ child_action ]                           # store path for later retrieval
            priority   = heuristic(state, problem) if callable(heuristic) else 0  # greedy breadth first search
            frontier.push( (child_state, child_path), priority )

            ### breadthFirstSearch() can terminate early with shortest path when ignoring path cost
            ### BUGFIX: terminating early will break autograder unit tests
            # if problem.isGoalState(child_state):
            #     return child_path
    else:
        return []  # breadthFirstSearch() is unsolvable



# python pacman.py -l tinyMaze   -p SearchAgent -a fn=gdfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=gdfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=gdfs
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=gbfs -l
# Mazes: bigMaze contoursMaze mediumDottedMaze mediumMaze mediumScaryMaze openMaze smallMaze testMaze tinyMaze
def greedyBreadthFirstSearch(
        problem: SearchProblem
) -> Union[List[str], bool]:
    return breadthFirstSearch(
        problem   = problem,
        heuristic = distanceHeuristic
    )


# python autograder.py -q q3
# python pacman.py -l mediumMaze       -p SearchAgent -a fn=ucs
# python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
# python pacman.py -l mediumScaryMaze  -p StayWestSearchAgent
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=ucs -l
def uniformCostSearch(
        problem:   SearchProblem,
        heuristic: Callable[ [Union[str,Tuple[int]], SearchProblem], int] = None,
) -> List[str]:
    """Search the node of least total cost first."""
    state      = problem.getStartState()
    frontier   = util.PriorityQueue()
    visited    = { state: True }      # True once removed from the frontier
    path_costs = { state: (0, []) }   # cost, path_actions

    frontier.push(state, 0)           # initialize root node, and add as first item in the queue

    loop = 0
    while not frontier.isEmpty():
        loop += 1
        state = frontier.pop()        # inspect next shortest path
        visited[state] = True         # only mark as visited once removed from the frontier
        (cost, action_path) = path_costs[state]

        # Uniform cost search must wait for node to be removed from frontier to guarantee least cost
        if problem.isGoalState(state):
            return action_path

        successors = problem.getSuccessors(state)

        ### python ./pacman.py -p SearchAgent -a fn=astar,prob=CornersProblem,heuristic=cornersHeuristicNull -q -l bigCorners
        ### Nodes Expanded | default=7949 | reversed %2 = 7907 | random.shuffle = 7869-7949
        if loop %2: successors = reversed(successors)      # move semi-diagonally in open spaces
        # random.shuffle(successors)                       # move semi-diagonally in open spaces

        for (child_state, child_action, edge_cost) in successors:
            if child_state in visited: continue              # avoid searching already explored states

            child_path      = action_path + [ child_action ]
            child_path_cost = cost + edge_cost
            heuristic_cost  = child_path_cost + (heuristic(child_state, problem) if callable(heuristic) else 0)

            # Check to see if an existing path to this node has already been found
            # Compare actual costs and not heuristic costs - which may be different
            if child_state in path_costs:
                prev_cost, prev_path = path_costs[child_state]
                if prev_cost <= child_path_cost:
                    continue  # child is worst than existing, skip to next successor

            path_costs[child_state] = (child_path_cost, child_path)   # store path + cost in dict rather than Queue
            frontier.update( child_state, heuristic_cost )            # process frontier in heuristic_cost order
    else:
        return []  # uniformCostSearch() is unsolvable


# python pacman.py -l openMaze  -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -l
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return uniformCostSearch(
        problem   = problem,
        heuristic = heuristic,
    )


# Abbreviations
bfs   = breadthFirstSearch
gbfs  = greedyBreadthFirstSearch
dfs   = depthFirstSearch
gdfs  = greedyDepthFirstSearch
astar = aStarSearch
ucs   = uniformCostSearch
