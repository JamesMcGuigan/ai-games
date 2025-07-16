# https://grok.com/chat/08848f31-8405-49c7-b7d9-607277a6a232

import numpy as np
import pygame
import random
from ple import PLE
from ple.games.pixelcopter import Pixelcopter  # https://github.com/ntasfi/PyGame-Learning-Environment/tree/master/ple/games

# Initialize the game
game = Pixelcopter(width=256, height=256)
game.rng = np.random.RandomState(24)  # Set random seed for reproducibility
p = PLE(game, fps=30, display_screen=True)
p.init()

# Main game loop
while True:
    if p.game_over():
        p.reset_game()
    state  = p.getGameState()
    action = random.choice(p.getActionSet())  # Example: random press "up" (replace with RL agent)
    reward = p.act(action)
    pygame.display.update()