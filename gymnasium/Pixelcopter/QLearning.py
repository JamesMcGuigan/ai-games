# https://grok.com/chat/45566a90-b147-4f74-8547-1119a63c68b4

import numpy as np
import pygame
import random
from ple import PLE
from ple.games.pixelcopter import Pixelcopter

"""
Q-learning agent for the Pixelcopter game in PLE.
The agent learns to navigate a helicopter through gates by discretizing the state space
and updating a Q-table using the Q-learning algorithm.
"""

# Configuration for Q-learning hyperparameters
Q_CONFIG = {
    'learning_rate': 0.1,        # Step size for Q-value updates
    'discount_factor': 0.9,      # Weight of future rewards
    'epsilon': 1.0,              # Initial exploration rate
    'epsilon_decay': 0.995,      # Decay rate for exploration
    'min_epsilon': 0.1,          # Minimum exploration rate
    'episodes': 1000             # Number of training episodes
}

# Initialize game environment
env = Pixelcopter(width=256, height=256)
env.rng = np.random.RandomState(24)  # Set seed for reproducibility
ple_env = PLE(env, fps=30, display_screen=True)
ple_env.init()

# Discretize state space
player_y_bins = np.linspace(0, 256, 10)      # 10 bins for helicopter y-position
player_vel_bins = np.linspace(-10, 10, 5)    # 5 bins for helicopter velocity
gate_distance_bins = np.linspace(0, 256, 10)  # 10 bins for distance to next gate
gate_top_y_bins = np.linspace(0, 256, 10)    # 10 bins for top of next gate

# Initialize Q-table
state_space_shape = (len(player_y_bins), len(player_vel_bins), len(gate_distance_bins), len(gate_top_y_bins))
action_space = ple_env.getActionSet()  # [None, 119] where None=no action, 119=up ('w' key)
q_table = np.zeros(state_space_shape + (len(action_space),))  # Shape: (y_bins, vel_bins, dist_bins, gate_y_bins, actions)

def get_discrete_state(state, debug=False):
    """
    Convert continuous game state to discrete indices for Q-table lookup.

    Args:
        state (dict): Game state with keys like 'player_y', 'player_vel', etc.
        debug (bool): If True, print state keys once for debugging.

    Returns:
        tuple: Discrete state indices (player_y, player_vel, gate_dist, gate_top_y).

    Raises:
        KeyError: If required state keys are missing.
    """
    if debug and not hasattr(get_discrete_state, 'printed'):
        print("State keys:", state.keys())
        get_discrete_state.printed = True

    required_keys = ['player_y', 'player_vel', 'next_gate_dist_to_player', 'next_gate_block_top']
    if not all(key in state for key in required_keys):
        missing = [key for key in required_keys if key not in state]
        raise KeyError(f"Missing state keys: {missing}")

    player_y_index = np.digitize(state['player_y'], player_y_bins) - 1
    player_vel_index = np.digitize(state['player_vel'], player_vel_bins) - 1
    gate_dist_index = np.digitize(state['next_gate_dist_to_player'], gate_distance_bins) - 1
    gate_top_y_index = np.digitize(state['next_gate_block_top'], gate_top_y_bins) - 1

    return (player_y_index, player_vel_index, gate_dist_index, gate_top_y_index)

def choose_action(state_indices, q_table, action_space, epsilon):
    """
    Select an action using epsilon-greedy policy.

    Args:
        state_indices (tuple): Discrete state indices.
        q_table (np.ndarray): Q-value table.
        action_space (list): List of possible actions.
        epsilon (float): Exploration probability.

    Returns:
        action: Selected action (None or 119).
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(action_space)
    action_idx = np.argmax(q_table[state_indices])
    return action_space[action_idx]

def train_q_learning(env, q_table, config):
    """
    Train the Q-learning agent on the Pixelcopter environment.

    Args:
        env: PLE environment instance.
        q_table (np.ndarray): Q-value table to update.
        config (dict): Q-learning hyperparameters.
    """
    epsilon = config['epsilon']
    for episode in range(config['episodes']):
        env.reset_game()
        total_reward = 0

        while not env.game_over():
            # Get current state
            state = env.getGameState()
            state_indices = get_discrete_state(state, debug=True)

            # Choose action
            action = choose_action(state_indices, q_table, action_space, epsilon)

            # Take action and observe reward
            reward = env.act(action)
            total_reward += reward

            # Get next state
            new_state = env.getGameState()
            new_state_indices = get_discrete_state(new_state)

            # Update Q-table: Q(s,a) += alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
            action_idx = action_space.index(action)
            current_q = q_table[state_indices][action_idx]
            max_future_q = np.max(q_table[new_state_indices])
            new_q = current_q + config['learning_rate'] * (
                reward + config['discount_factor'] * max_future_q - current_q
            )
            q_table[state_indices][action_idx] = new_q

            pygame.display.update()

        # Decay exploration rate
        epsilon = max(config['min_epsilon'], epsilon * config['epsilon_decay'])
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Run training
train_q_learning(ple_env, q_table, Q_CONFIG)

# Save Q-table for future use
np.save("QLearning.npy", q_table)