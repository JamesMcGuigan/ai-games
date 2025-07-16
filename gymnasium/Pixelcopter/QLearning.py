# https://grok.com/chat/45566a90-b147-4f74-8547-1119a63c68b4

import numpy as np
import pygame
import random
from ple import PLE
from ple.games.pixelcopter import Pixelcopter

# Initialize the game
game = Pixelcopter(width=256, height=256)
game.rng = np.random.RandomState(24)
p = PLE(game, fps=30, display_screen=True)
p.init()

# Q-Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.1
episodes = 1000

# Discretize state space
y_bins = np.linspace(0, 256, 10)  # 10 bins for player_y
vel_bins = np.linspace(-10, 10, 5)  # 5 bins for velocity
dist_bins = np.linspace(0, 256, 10)  # 10 bins for distance to next gate
gate_y_bins = np.linspace(0, 256, 10)  # 10 bins for next_gate_block_top

# Initialize Q-table
state_space_size = (len(y_bins), len(vel_bins), len(dist_bins), len(gate_y_bins))
action_space = p.getActionSet()  # [None, 119]
q_table = np.zeros(state_space_size + (len(action_space),))


def get_discrete_state(state):
    # Print state keys to debug (only once)
    if not hasattr(get_discrete_state, 'printed'):
        print("State keys:", state.keys())
        get_discrete_state.printed = True

    player_y = np.digitize(state['player_y'], y_bins) - 1
    player_vel = np.digitize(state['player_vel'], vel_bins) - 1
    gate_dist = np.digitize(state['next_gate_dist_to_player'], dist_bins) - 1
    gate_y = np.digitize(state['next_gate_block_top'], gate_y_bins) - 1
    return (player_y, player_vel, gate_dist, gate_y)


# Main training loop
for episode in range(episodes):
    p.reset_game()
    total_reward = 0

    while not p.game_over():
        state = p.getGameState()
        discrete_state = get_discrete_state(state)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_space)
        else:
            action_idx = np.argmax(q_table[discrete_state])
            action = action_space[action_idx]

        # Take action
        reward = p.act(action)
        total_reward += reward

        # Get new state
        new_state = p.getGameState()
        new_discrete_state = get_discrete_state(new_state)

        # Update Q-table
        action_idx = action_space.index(action)
        current_q = q_table[discrete_state][action_idx]
        max_future_q = np.max(q_table[new_discrete_state])
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        q_table[discrete_state][action_idx] = new_q

        pygame.display.update()

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save Q-table (optional)
np.save("QLearning.npy", q_table)