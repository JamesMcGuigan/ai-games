# https://grok.com/chat/91aceab9-0be9-4165-ba4b-697f4a642787

# LunarLander-v2 with DQN (Deep Q-Network)
# Goal: Train to land softly (reward > 200 average)
# DOCS: https://gymnasium.farama.org/environments/box2d/lunar_lander/
# Based on standard DQN implementation for discrete actions

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random

# Hyperparameters
GAMMA = 0.99          # Discount factor
EPS_START = 1.0       # Initial epsilon for exploration
EPS_END = 0.01        # Final epsilon
EPS_DECAY = 0.995     # Epsilon decay rate
BATCH_SIZE = 128      # Replay batch size
LR = 0.001            # Learning rate
REPLAY_SIZE = 10000   # Replay buffer size
TARGET_UPDATE = 10    # Target net update frequency
NUM_EPISODES = 1000   # Training episodes
SEED = 42

# Q-Network (MLP)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Training Function
def train_dqn():
    env = gym.make("LunarLander-v3")
    env.reset(seed=SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    state_size = env.observation_space.shape[0]  # 8
    action_size = env.action_space.n             # 4

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    epsilon = EPS_START
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # Train if buffer is full enough
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Compute Q values
                current_q = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                # Loss and update
                loss = nn.functional.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon:.3f}")

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Average reward (last 100 eps): {avg_reward}")

    env.close()
    torch.save(policy_net.state_dict(), "DQNModule.pth")
    return policy_net

# Evaluation Function (Load and Run)
def evaluate_dqn(policy_net):
    env = gym.make("LunarLander-v3", render_mode="human")
    state, _ = env.reset(seed=SEED)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = next_state

    print(f"Evaluation: Total reward = {total_reward}, Episode length = {steps} steps")
    env.close()

# Run training and evaluation
policy_net = train_dqn()
evaluate_dqn(policy_net)