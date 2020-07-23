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

This is an object oriented implementation of the MiniMax with AlphaBeta pruning algorithm and iterative deepening 
using a custom heuristic.

This implementation converts the board into a numpy array. 
Whilst convenient for programming, this might be slower (due to object copying) than performing operations 
directly on list primitives. [Peter Cnudde (1st in this Competition)](https://www.kaggle.com/c/connectx/discussion/158190)
claims that bitboards are the fastest internal representation to work and capable of reaching depth 12-16.

## Timeouts

Attempting to use signals ([util/call_with_timeout_ms.py](util/call_with_timeout_ms.py)) 
to throw an exception on timeout didn't seem to work when run inside Kaggle submit. 

The fastest way to exit a recursive loop is by throwing an exception. 

Loop exit times are usually in the 0.1-3ms range, but on rare occasions the code will take longer than 1000ms+.
There can be upto 49 turns in a game, so a Timeout on any of these will result in total game loss with an Err score.

Empirical observation of the Kaggle Leaderboard results for the effect of `safety_time` values:
- 0.75s = kaggle error rate of 26/154 = 15% 
- 1.15s = kaggle error rate of 16/360 = 4.5% 

The root cause of these random timeouts was discovered as the garbage collector. This can be solved
by running `gc.disable()` as the first action of the agent, then `gc.enable()` to schedule garbage collection
during our opponents turn (or the inter-turn period). 

With the gc disabled, `safety_time=10ms` is reliable on localhost but fails on Kaggle Submit, 
so currently playing safe with `safety_time=1500ms`.


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

## AlphaBeta Agent Bitboard

- [agents/AlphaBetaAgent/AlphaBetaBitboard.py](agents/AlphaBetaAgent/AlphaBetaBitboard.py)

Focusing attention back on raw performance resulted in a bitboard implementation. 
A 84 bit number was used, divided into 2 x 42 bit subsections. 
The first half the bitboard stored if a square was empty or contained a piece (0==empty, 1==filled).
The second half the bitboard stored which player's token was in each square (0==p1, 1==p2).

The gameover/utility function was implemented by creating bitmasks for all 69 possible win lines,
for fast iteration to see if all squares where filled, and then if so where they all filled by the same player. 

A simple but fast heuristic was designed that emulated the gameover/utility methodology, 
but also matched on empty squares, signifying the number of potential connect4 lines that could be created.

The pure python implementation (without numba) was able to get to depth=7 on the first move, 
climbing to depth=12 during the midgame, compared with depth=5-7 for the object oriented 
AlphaBetaAgent. This translates to a depth advantage of 2-5. 

AlphaBetaBitboard is able to beat the depth=4 Kaggle Negamax agent quite easily. 

AlphaBetaBitboard with the original first-pass bitboard heuristic scored a 55% winrate against 
object-oriented AlphaBetaAgent with its slower (but stronger) LibertiesHeuristic. 
A small improvement using  `math.log2() % 1 == 0` to heavily discount single square lines improved the winrate to 70%.
Adding in a bonus double_attack_score for overlapping connect4 lines reduced the score at first to 55% 
with a large percentage of draws. This was tweaked to only include connect4s that overlaped via a single square 
(mostly those pointing in different directions) which boosted the winrate to 80%.
Hyperparemeter optimization resulting in `double_attack_score=0.5` finally achieved a 100% winrate.

Attempting to optimise AlphaBetaBitboard with numba did not result in any significant performance increase.


# Future Ideas
- Create an opening book
- Create an endgames table
- Ant Colony / Monty Carlo Agent
- Deep learning / Decision Tree / XGBoost 
  - As a heuristic to predict winning positions
  - As an agent to predict the next move
  - In a GAN style league to train against other Agents