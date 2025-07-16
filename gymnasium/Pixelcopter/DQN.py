# https://grok.com/chat/45566a90-b147-4f74-8547-1119a63c68b4

import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from ple import PLE
from ple.games.pixelcopter import Pixelcopter

"""
Deep Q-Learning (DQN) agent for the Pixelcopter game in PLE.
The agent uses a neural network to approximate Q-values, trained with experience replay
and a target network to navigate the helicopter through gates.
"""

# Configuration for DQN hyperparameters
DQN_CONFIG = {
    'learning_rate': 0.001,      # Learning rate for optimizer
    'discount_factor': 0.9,      # Weight of future rewards
    'epsilon': 1.0,              # Initial exploration rate
    'epsilon_decay': 0.995,      # Decay rate for exploration
    'min_epsilon': 0.1,          # Minimum exploration rate
    'episodes': 1000,            # Number of training episodes
    'replay_buffer_size': 10000, # Size of experience replay buffer
    'batch_size': 32,            # Batch size for training
    'target_update_freq': 100    # Steps between target network updates
}

# Neural network for Q-value approximation
class QNetwork(nn.Module):
    """Neural network to approximate Q-values: state -> Q-values for each action."""
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Initialize game environment
env = Pixelcopter(width=256, height=256)
env.rng = np.random.RandomState(24)  # Set seed for reproducibility
ple_env = PLE(env, fps=30, display_screen=True)
ple_env.init()

# Initialize DQN components
state_dim = 4  # State: [player_y, player_vel, next_gate_dist_to_player, next_gate_block_top]
action_space = ple_env.getActionSet()  # [None, 119] where None=no action, 119=up ('w' key)
action_dim = len(action_space)
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())  # Copy weights to target network
optimizer = optim.Adam(q_network.parameters(), lr=DQN_CONFIG['learning_rate'])
replay_buffer = deque(maxlen=DQN_CONFIG['replay_buffer_size'])

def get_state_vector(state: dict, debug: bool = False) -> np.ndarray:
    """
    Extract relevant state features into a vector.

    Args:
        state: Game state dictionary with keys like 'player_y', 'player_vel', etc.
        debug: If True, print state keys once for debugging.

    Returns:
        np.ndarray: State vector [player_y, player_vel, next_gate_dist_to_player, next_gate_block_top].

    Raises:
        KeyError: If required state keys are missing.
    """
    if debug and not hasattr(get_state_vector, 'printed'):
        print("State keys:", state.keys())
        get_state_vector.printed = True

    required_keys = ['player_y', 'player_vel', 'next_gate_dist_to_player', 'next_gate_block_top']
    if not all(key in state for key in required_keys):
        missing = [key for key in required_keys if key not in state]
        raise KeyError(f"Missing state keys: {missing}")

    return np.array([
        state['player_y'],
        state['player_vel'],
        state['next_gate_dist_to_player'],
        state['next_gate_block_top']
    ], dtype=np.float32)

def choose_action(state: np.ndarray, q_network: QNetwork, action_space: list, epsilon: float):
    """
    Select an action using epsilon-greedy policy.

    Args:
        state: State vector.
        q_network: Neural network for Q-value prediction.
        action_space: List of possible actions.
        epsilon: Exploration probability.

    Returns:
        action: Selected action (None or 119).
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(action_space)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = q_network(state_tensor)
    action_idx = torch.argmax(q_values, dim=1).item()
    return action_space[action_idx]

def train_dqn(env: PLE, q_network: QNetwork, target_network: QNetwork, config: dict) -> list:
    """
    Train the DQN agent on the Pixelcopter environment.

    Args:
        env: PLE environment instance.
        q_network: Main Q-network for training.
        target_network: Target network for stable Q-value targets.
        config: DQN hyperparameters.

    Returns:
        list: Total rewards per episode for monitoring.
    """
    epsilon = config['epsilon']
    step_count = 0
    episode_rewards = []  # Track rewards for monitoring

    for episode in range(config['episodes']):
        env.reset_game()
        total_reward = 0.0  # Ensure float for accumulation
        state = get_state_vector(env.getGameState(), debug=True)

        while not env.game_over():
            # Choose and take action
            action = choose_action(state, q_network, action_space, epsilon)
            reward = env.act(action)
            total_reward += float(reward)  # Convert reward to float
            next_state = get_state_vector(env.getGameState())
            done = env.game_over()

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            step_count += 1

            # Train network if buffer has enough samples
            if len(replay_buffer) >= config['batch_size']:
                batch = random.sample(replay_buffer, config['batch_size'])
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to NumPy arrays before tensors for efficiency
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor([action_space.index(a) for a in actions])
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))

                # Compute Q-values and targets
                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    targets = rewards + config['discount_factor'] * next_q_values * (1 - dones)

                # Update network
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network periodically
            if step_count % config['target_update_freq'] == 0:
                target_network.load_state_dict(q_network.state_dict())

            pygame.display.update()

        # Decay exploration rate
        epsilon = max(config['min_epsilon'], epsilon * config['epsilon_decay'])
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

    return episode_rewards

# Run training and save rewards
episode_rewards = train_dqn(ple_env, q_network, target_network, DQN_CONFIG)

# Save Q-network weights
torch.save(q_network.state_dict(), "DQN.pth")

# Save rewards for analysis
np.save("DQN.npy", np.array(episode_rewards))