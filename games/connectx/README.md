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


# Minimax, Negamax and AlphaBeta

## AlphaBeta Agent 

Leaderboard Score: **1040 (top 8%)** | [AlphaBetaAgent + LibertiesHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16236086)

- [agents/AlphaBetaAgent/AlphaBetaAgent.py](agents/AlphaBetaAgent/AlphaBetaAgent.py)
- [kaggle:connectx-alphabeta-minimax-with-numba](https://www.kaggle.com/jamesmcguigan/connectx-alphabeta-minimax-with-numba)

This is an object oriented implementation of the MiniMax with AlphaBeta pruning algorithm and iterative deepening 
using a custom heuristic.

This implementation converts the board into a numpy array. 
Whilst convenient for programming, this might be slower (due to object copying) than performing operations 
directly on list primitives. [Peter Cnudde (1st in this Competition)](https://www.kaggle.com/c/connectx/discussion/158190)
claims that bitboards are the fastest internal representation to work and capable of reaching depth 12-16.

### Timeouts

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
so currently playing safe with `safety_time=2000ms`.



## AlphaBeta Bitboard
Leaderboard Score: **1080 (top 5%)** | [AlphaBetaAgentBitboard + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16618373)

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
with a large percentage of draws. This was tweaked to only include connect4s that overlapped via a single square 
(mostly those pointing in different directions) which boosted the winrate to 80%.
Hyperparameter optimization resulting in `double_attack_score=0.5` finally achieved a 100% winrate.

Attempting to optimise AlphaBetaBitboard with numba did not result in any significant performance increase.



## Negamax Bitboard

Leaderboard Score: **1010 (top 10%)** | [Negamax + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16765616)

- [agents/Negamax/Negamax.py](agents/Negamax/Negamax.py)

Negamax is similar conceptually to Minimax, but has a single recursive function. 
This implementation was written in a functional rather than OO style for numba @njit optimization but without caching.

I still have not managed to get numba working inside kaggle submit, so this actually runs slower than the cached 
version of AlphaBeta. 

[Unit tests](tests/test_board_positions.py) for board positions highlighted that the `bitboard_gameovers_heuristic()`
was not an admissible heuristic, and thus failing to see some forced moves due to AlphaBeta pruning, so this was disabled.



## Minimax Bitboard

Leaderboard Score: **1030 (top 8%)** | [Negamax + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16765616)

- [agents/AlphaBetaAgent/MinimaxBitboard.py](agents/AlphaBetaAgent/MinimaxBitboard.py)

This was a version of AlphaBeta Bitboard with alpha beta pruning disabled, inspired by the unit test failures for Negamax.

However this approach scores worse on the leaderboard compared to the version with alpha-beta pruning enabled.
This either indicates that greater depth is still superior to accuracy, or else there is a bug in the Negamax 
alpha-pruning code.



# Monty Carlo Tree Search

## Pure Monty Carlo Search
Leaderboard Scores: 
- **1075 (top 6%)** | [MontyCarloLinkedList + Cached Data](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16816641)
- **1065 (top 7%)** | [MontyCarloLinkedList](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16779472)

Source:
- [agents/MontyCarlo/MontyCarloLinkedList.py](agents/MontyCarlo/MontyCarloLinkedList.py)

Whereas Minimax / Negamax performs a breadth first search of the game state tree, Monty Carlo selectively expands
the tree deeper in areas where success is more probable. Tree/graph expansion follows a similar shape to A* search. 

The algorithm starts with a root node with unexpanded children.

Expansion happens in two parts. First is node selection, where is the tree is traversed from the root node, 
choosing the child node with the highest UCB score, which includes an additional term to encourage exploration.
When an unexpanded leaf node is reached, a random simulation of the game is run from that position 
with the score and total counts backpropergated along the tree path.

Expansions are repeatedly run until the timeout expires, and the child of root node with the highest score is chosen 
as the returned as the agent action.


## Monty Carlo Heuristic Search

Leaderboard Scores: 
- **1110 (top 4%)** | [MontyCarloHeuristic + BitboardGameoversHeuristic + Cached Data](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16832763)
- **1110 (top 4%)** | [MontyCarloHeuristic + BitboardGameoversHeuristic + Cached Data](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16832763)
- **1070 (top 6%)** | [MontyCarloHeuristic + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16783457)

Source:
- [agents/MontyCarlo/MontyCarloLinkedList.py](agents/MontyCarlo/MontyCarloLinkedList.py)

This approach replaces random simulation (producing an integer score of 0|1) with a sigmoided version of 
`bitboard_gameovers_heuristic()` (producing a floating point score between 0 and 1). 

Scaling the sigmoid by factor of 6 (division), produced the highest winrate compared to other numbers. 
This means a heuristic value of +6 would return a sigmoid score of +0.73, which is pretty winning. 
Smaller differences in heuristic score would result in a value much closer to 0.5 draw. 
If a terminal state in the game tree is reached, an integer score of 0|1 is returned.

Compared to random simulation, a heuristic provides more indepth knowledge about a position, and it is significantly 
faster to compute than running a full game simulation. This means many more expansions can be run within the same time limit.


## Ant Colony Heuristic Search
Leaderboard Score: **920 (top 18%)** | [AntColonyTreeSearch + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16800699)

- [agents/MontyCarlo/AntColonyTreeSearch.py](agents/MontyCarlo/AntColonyTreeSearch.py)
- [agents/broken/AntColonyTreeSearch.py](agents/broken/AntColonyTreeSearch.py)

This is a experimental variation of the Monty Carlo tree search, but instead of using `max(UCB_score)`,
the score is treated as a pheromone trail and weighted random selection is used.

Maybe this approach needs further experimentation to get right, but currently it produces results significantly 
worse than Pure Monty Carlo and Minimax/Negamax. 

Ideas:
- remove the distinction between path selection and simulation, and follow the pheromones trail until terminal state
- store pheromones in a dictionary using mirror hash
- leave an additional strong pheromone trail recording actual game result.
- have the ant colony play continually play against itself with a very short timeout   
- decay old pheromones as a moving average


### Cached Data
- [util/base64_file.py](util/base64_file.py)
- [core/PersistentCacheAgent.py](core/PersistentCacheAgent.py)

Much of the MontyCarlo runtime is spent expanding the tree and computing simulations/heuristics. 
Some of this could be precomputed to extend the depth of search possible during the 8 second turn timer.

The state tree needed to be persisted to disk, in a pickle.zip.base64 format suitable for embedding as text 
in a python script, then reloaded upon initialization. 

Several hours of localhost runtime where spent playing the MontyCarlo agents against themselves and other agents
including hundreds of matches against random_agent. This effectively generated an opening book and precached engdame 
values for some expected lines of play.  

In theory, kaggle allows 100Mb of submission data, but in practice data files larger than 5-10Mb 
cause kaggle submission errors. The original datafile generated by the above process was 47Mb which was too large.
By pruning the tree of any nodes that had not been fully expanded, the filesize was reduced to a workable 5Mb.   

This cached datafile significantly improved winrates on the leaderboard, both for Random Simulation 
and Heuristic versions of Monty Carlo Tree Search.


# Heuristics

## Liberties Heuristic

Leaderboard Score: **1040 (top 8%)** | [AlphaBetaAgent + LibertiesHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16236086)

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


## Bitboard Gameovers Heuristic 

Leaderboard Scores: 
- **1110 (top 4%)** | [MontyCarloHeuristic + BitboardGameoversHeuristic + Cached Data](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16832763)
- **1080 (top 5%)** | [AlphaBetaAgentBitboard + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16618373)
- **1070 (top 5%)** | [MontyCarloHeuristic + BitboardGameoversHeuristic](https://www.kaggle.com/c/connectx/submissions?dialog=episodes-submission-16783457)


Code:
- [heuristics/BitboardHeuristic.py](heuristics/BitboardGameoversHeuristic.py)

This was a first attempt to write a performance optimized heuristic using vectorized bitshifting

The heuristic, as the name suggests, is based on the the `gameovers()` function which has a precomputed array 
of bitmasks for all possible connect 4 locations, then tests to see if the player pieces have filled in all those bits.

The heuristic presents a simplified problem by counting the number of possible connect 4 lines remaining by including
empty squares when comparing against the bitmask. A bitmask containing only a single bit can be quickly detected using  
`log2() % 1 == 0`. 

Lines containing only a single player piece are discounted with a score of 0.1, compared to 1 for a multibit mask of player piece. 

An additional `double_attack_score` of 0.5 is given if any two gameover bitmasks happen to overlap by a single square.
This does not offer true double attack detection, but it rewards designing intersecting lines that are 
somewhat correlated with double attacks. In practice it improved the winrate vs AlphaBetaAgent with LibertiesHeuristic 
from 50% to near 100%.

The heuristic returns the difference between the players bitmask count and the opponents bitmask count.


# Numba 

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

Memoization in numba may be possible by using a int64 memory address pointer
- https://stackoverflow.com/questions/63107727/numba-compatable-memoization
- https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function/61550054#61550054

Work in progress: create testcases for which @njit functionality works inside kaggle submit
- https://www.kaggle.com/jamesmcguigan/connectx-numba-testcases


# Tests
## Unit Tests
- [./heuristics/LibertiesHeuristic_test.py](./heuristics/LibertiesHeuristic_test.py)
- [./core/ConnectXBBN_test.py](./core/ConnectXBBN_test.py)
- [./core/ConnectX_test.py](./core/ConnectX_test.py)
- [./core/ConnectXBitboard_test.py](./core/ConnectXBitboard_test.py)
- [./util/base64_file_test.py](./util/base64_file_test.py)

Unit tests validate that individual functions work as expected.


## Integration Tests
- [tests/test_board_positions.py](tests/test_board_positions.py)
- [tests/test_can_win_against.py](tests/test_can_win_against.py)

Integration testing can be applied to AI algorithms by giving them game puzzles to solve, 
especially in positions where a human can verify that there is only one (or two) winning/losing moves.

The simplest is being one move away from connect 4 and seeing if the agent can either find the winning move, 
or block the opponent from that square. More complicated positions include being able to spot a double attack, 
which requires a Minimax search depth of 4, or knowing which column to play during an endgame. 

A second form of integration tests is a live matchup against the inbuilt kaggle agents: random and negamax.
Any leaderboard worthy agent should be able score a near 100% winrate against these opponents, 
so a logic mistake in the algorithm will show up as a test failure.


# Future Ideas
- Create an opening book for Minimax
- Create an endgames table
- Deep learning / Decision Tree / XGBoost 
  - As a heuristic to predict winning positions
  - As an agent to predict the next move
  - In a GAN style league to train against other Agents
- Add table of contents to README using [asciidoc](https://gist.github.com/dcode/0cfbf2699a1fe9b46ff04c41721dda74) 