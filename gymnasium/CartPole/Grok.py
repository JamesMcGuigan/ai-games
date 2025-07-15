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
        self.fc1 = nn.Linear(state_size, 64)  # Increased neurons
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 64  # Increased batch size
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99  # Faster decay
max_episodes = 1000
target_update = 10  # More frequent updates
max_steps = 500

# Initialize
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=buffer_size)
criterion = nn.MSELoss()
episode_rewards = deque(maxlen=100)
losses = []

# Training Loop
for episode in range(max_episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    total_reward = 0
    step = 0

    for t in range(max_steps):
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Add batch dimension
        total_reward += reward

        # Store transition
        memory.append((state, action, reward, next_state, done))
        state = next_state
        step += 1

        # Train on mini-batch
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states)  # Stack states
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.cat(next_states)
            dones = torch.FloatTensor(dones)

            # Compute Q-values
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0].detach()
            targets = rewards + gamma * next_q_values * (1 - dones)

            # Update network
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            losses.append(loss.item())

        if done or truncated:
            break

    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Logging
    avg_loss = np.mean(losses[-100:]) if losses else 0
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

    # Check if solved
    if len(episode_rewards) >= 100:
        avg_reward = np.mean(episode_rewards)
        if avg_reward >= 195:
            print(f"CartPole solved! Average reward: {avg_reward:.2f} over last 100 episodes")
            break

env.close()

# Optional: Plot rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards")
plt.show()