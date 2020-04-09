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
        ) -> Union[List[str], bool]:
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
            if next_actions == False:  # we have hit a dead end
                continue               # try the next available path
            else:
                return next_actions    # return and save results

        ### if len(successors) == 0 or all successors returned false
        return False                   # backtrack



# python pacman.py -l tinyMaze   -p SearchAgent -a fn=gdfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=gdfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=gdfs
# find layouts -name '*Maze*' | grep -v Dotted | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=gdfs -l
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
        problem:   SearchProblem,
        heuristic: Callable[ [Union[str,Tuple[int]], SearchProblem], int] = None
) -> Union[List[str], bool]:
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

            visited[child_state] = True                                            # mark as seen
            child_path  = action_path + [ child_action ]                           # store path for later retrieval
            child_cost += heuristic(state, problem) if callable(heuristic) else 0  # greedy breadth first search
            frontier.push( (child_state, child_path), child_cost )
    else:
        return False  # breadthFirstSearch() is unsolvable



# python pacman.py -l tinyMaze   -p SearchAgent -a fn=gdfs
# python pacman.py -l mediumMaze -p SearchAgent -a fn=gdfs
# python pacman.py -l bigMaze    -p SearchAgent -a fn=gdfs
# find layouts -name '*Maze*' | grep -v Dotted | perl -p -e 's!^.*/|\..*$!!g' | xargs -t -L1 python pacman.py -p SearchAgent -a fn=gbfs -l
# Mazes: bigMaze contoursMaze mediumDottedMaze mediumMaze mediumScaryMaze openMaze smallMaze testMaze tinyMaze
def greedyBreadthFirstSearch(
        problem: SearchProblem,
) -> Union[List[str], bool]:
    return breadthFirstSearch(
        problem   = problem,
        heuristic = distanceHeuristic
    )



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs   = breadthFirstSearch
gbfs  = greedyBreadthFirstSearch
dfs   = depthFirstSearch
gdfs  = greedyDepthFirstSearch
astar = aStarSearch
ucs   = uniformCostSearch
