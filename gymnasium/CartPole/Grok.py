# https://grok.com/chat/6430c86d-53c9-4870-ba7e-f0f5989fc144
# DOCS: https://gymnasium.farama.org/environments/classic_control/cart_pole/

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n  # 2


# Neural Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameters
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 32
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
max_episodes = 1000
target_update = 100

# Initialize
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=buffer_size)
criterion = nn.MSELoss()

# Training Loop
for episode in range(max_episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0

    for t in range(500):  # Max steps per episode
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(next_state)
        total_reward += reward

        # Store transition
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Train on mini-batch
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.FloatTensor(dones)

            # Compute Q-values
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

            # Update network
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}")

    # Check if solved (average reward > 195 over 100 episodes)
    if episode > 100:
        avg_reward = np.mean([memory[i][2] for i in range(len(memory)) if memory[i][2] for _ in range(100)])
        if avg_reward > 195:
            print("CartPole solved! ", avg_reward)
            break

env.close()


# Evaluation Function (Load and Run) - Also GPU-enabled
def evaluate_dqn(policy_net):
    env = gym.make("LunarLander-v3", render_mode="human")
    state, _ = env.reset(seed=SEED)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # To device
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
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)  # To device
        state = next_state

    print(f"Evaluation: Total reward = {total_reward}, Episode length = {steps} steps")
    env.close()
