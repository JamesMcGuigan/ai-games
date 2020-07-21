# Connect X / Connect 4

This project explores various approaches to solving the classic Connect 4 game on [Kaggle](https://www.kaggle.com/c/connectx)



# Random Agent

Leaderboard Score: **360-415**
- [agents/RandomAgent/RandomAgent.py](agents/RandomAgent/RandomAgent.py)

This is the simplest agent possible. Validate the list of legal moves, and then pick one at random. 

This agent is very fast, but mainly provides a baseline score for other agents.




# Rules Agent 
Leaderboard Score: **424**

- [agents/RulesAgent/RulesAgent.py](agents/RulesAgent/RulesAgent.py)
- [kaggle:connectx-manually-coded-strategies](https://www.kaggle.com/jamesmcguigan/connectx-manually-coded-strategies?scriptVersionId=35203084)

This was a first attempt to create a simple agent with manually defined rules.

Strategies included:
- Always play middle column if no other columns have been played
- Look for an immediately winning move, then play it
- If the opponent has a winning move next turn, then block it.
- Look for possible double attacks either 2 or 3 moves away
- Else extend the longest existing line
 
The work here included exploration of how to manipulate the Kaggle data structures, 
implementing base classes required for playing the game, and this also provided inspiration for future heuristics.
The implementation of these rules was not thoroughly tested and may contain bugs. 

This agent is also very quick, but doesn't look very far ahead. 
Scores slightly better than the range for RandomAgents, 
but this approach is not highly competitive on the leaderboard.



# AlphaBeta Agent 
Leaderboard Score: **1033.0 (top 8%)** with Liberties Heuristic

- [agents/AlphaBetaAgent/AlphaBetaAgent.py](agents/AlphaBetaAgent/AlphaBetaAgent.py)
- [kaggle:connectx-alphabeta-minimax-with-numba](https://www.kaggle.com/jamesmcguigan/connectx-alphabeta-minimax-with-numba)

This is an object oriented implementation of the AlphaBeta algorithm with iterative deepening using a custom heuristic.

This implementation converts the board into a numpy array. 
Whilst convenient for programming, this might be slower (due to object copying) than performing operations 
directly on list primitives. [Peter Cnudde (1st in this Competition)](https://www.kaggle.com/c/connectx/discussion/158190)
claims that bitboards are the fastest internal representation to work and capable of reaching depth 12-16.

## Timeouts

Attempting to use signals ([util/call_with_timeout_ms.py](util/call_with_timeout_ms.py)) 
to throw an exception on timeout didn't seem to work when run inside Kaggle submit. 

The fastest way to exit a recursive loop is by throwing an exception. 

Usually this can be done in under 3ms, but on rare occasions the code will take longer than 1000ms+.
There can be upto 49 turns in a game, and a Timeout on any of them will result in total game loss with an Err score.

Empirical observation of the Kaggle Leaderboard results for the effect of `safety_time` values:
- 0.75s = kaggle error rate of 26/154 = 15% 
- 1.15s = kaggle error rate of 16/360 = 4.5% 

Given the high observed error rate, and exponential time complexity of iterative deepening, 
overall score is probably improved more with a generous `safety_time`. 
On an empty board, depth 5 is reached in `2.28s` and depth 6 in `9.25s` with a default timeout of 8 seconds. 
This squeezing in an extra second of compute will generally not result in an extra level of depth being returned.

## Numba 

An attempt was made refactor using [numba](https://numba.pydata.org/) `@njit` for performance optimization. 
This probably would have been eaiser to do had the code been written designed with numba in mind, as `@njit`
requires `forceobj=True` when accessing object oriented instances, which removes most of the performance gains 
and can even make things slower.

Only a small number of functions in [LibertiesHeuristic.py](heuristics/LibertiesHeuristic.py) where 
able to be optimized this way, and still work upon kaggle submit, resulting in maybe a 10% performance improvement
in [benchmarks](agents/AlphaBetaAgent/AlphaBetaAgent.benchmark.py). 

Numba Notes:
- numba doesn't like object oriented programming or `frozenset()`
- `@jitclass` doesn't like class properties or `@cached_property`
- numba sometimes requires rewriting code such as list comprehensions explicitly 
- `cache=True` doesn't work for Kaggle submit due to a read-only filesystem. 
- `@numba.generated_jit(nopython=True, nogil=True, cache=False, forceobj=False, parallel=True)` 
has been shown to work on kaggle submit, which triggers a compile at startup rather than runtime


## Liberties Heuristic
- [heuristics/LibertiesHeuristic.py](heuristics/LibertiesHeuristic.py)

This was a first attempt at writing a heuristic function for AlphaBeta.

It iterates over the cells of board, detecting lines still capable of making a connect 4 and  
then counting the number of liberties/extensions available for each line using the formula:
```
return np.sum([ len(extension)**1.25 for extension in self.extensions ]) / ( self.game.inarow**2 )
```

This doesn't take into account the odd/even height of available liberties. 
The implementation is not perfectly efficient and can't be Numba'ed easily. 

Occasionally the heuristic does seem to miss obvious blocking moves against an opponent who is able to make 
Connect4 on the next move.

The profiler reports the heuristic accounts for 93% of runtime which is overly expensive and AlphaBeta 
is usually only able to reach depth 5 using this heuristic.

Even still AlphaBeta + LibertiesHeuristic does score in the top 8% Kaggle Leaderboard
