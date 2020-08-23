# Connect 4 Neural Networks

This is a first attempt at building a neural network in pytorch for Connect 4 functions.

Exploration will start with modelling simple functions, to determine the network sizes and architectures,
then later attempt to design a neural network heuristic function, which can be combined with 
either Minimax or MCTS
 
  
## is_gameover()

This is the first test, which will attempt to get 100% accuracy to model the 
function: `is_gameover(bitboard: np.ndarray) -> bool`

