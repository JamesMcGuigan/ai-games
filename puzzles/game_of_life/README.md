# Reverse Game of Life - Z3 Constraint Satisfaction

Kaggle Notebooks:
- https://www.kaggle.com/jamesmcguigan/game-of-life-z3-constraint-satisfaction/

---

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is an example of 2D cellular automata. 

I have previously written an interactive playable demo of the forward version of this game:
- https://life.jamesmcguigan.com/

Using the classic ruleset on a 25x25 board with wraparound, the game evolves at each timestep according to the following rules
- Overpopulation: if a living cell is surrounded by more than three living cells, it dies.
- Stasis: if a living cell is surrounded by two or three living cells, it survives.
- Underpopulation: if a living cell is surrounded by fewer than two living cells, it dies.
- Reproduction: if a dead cell is surrounded by exactly three cells, it becomes a live cell.

Or expressed algebraically:
- living + 4-8 neighbours = dies
- living + 2-3 neighbours = lives
- living + 0-1 neighbour  = dies
- dead   +   3 neighbours = lives


To reverse the arrow of time:
- any living cell must have had living 2-3 neighbours in the previous timestep
- any dead cell must have had either 0-1 or 4-8 neighbours in the previous timestep
- any dead cell with distance of greater than 2 from a living cell can be ignored and assumed to have 0 neighbours
  - there are a near infinite number of self-contained patterns could have been born and died out in empty space
  - however, for the sake of the competition, ignoring them will greatly reduce the search space


Whilst there have been many proposed solutions involving CNN neural networks (when all you have is a hammer, everything looks like a nail), this is in fact a classic constraint satisfaction problem. Here are some previous examples of using the Z3 library: 
- https://www.kaggle.com/jamesmcguigan/z3-sudoku-solver
- https://www.kaggle.com/jamesmcguigan/cryptarithmetic-solver