# https://claude.ai/chat/c794f10a-5aea-4291-af12-7c9be92de55c
# https://grok.com/chat/b8b0e4c0-e37f-4b6e-9af3-72148eb9b097
# https://chatgpt.com/c/687bba22-130c-8004-8376-c697ce4da916
# https://gemini.google.com/app/402d034de62a96c0 - cleanest code

import gymnasium as gym
import numpy as np
from collections import defaultdict

# 1. Initialize environment
env = gym.make('CliffWalking-v0', is_slippery=False)

# 2. Define hyperparameters
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 1.0     # Exploration-exploitation trade-off (starts high for exploration)
epsilon_decay_rate = 0.001
min_epsilon = 0.01
num_episodes = 2000

# Initialize Q-table
# A dictionary can be used for sparse state spaces, but for CliffWalking's discrete states, a NumPy array is fine.
Q = np.zeros((env.observation_space.n, env.action_space.n))

# To store episode rewards and lengths for plotting
episode_rewards = []
episode_lengths = []

# Function to choose action using epsilon-greedy policy
def epsilon_greedy_action(state, Q_table, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore: choose a random action
    else:
        return np.argmax(Q_table[state, :]) # Exploit: choose best known action

# 3. Training Loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_episode_reward = 0
    steps_in_episode = 0

    while not done:
        action = epsilon_greedy_action(state, Q, epsilon, env.action_space.n)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update rule
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state
        total_episode_reward += reward
        steps_in_episode += 1

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

    episode_rewards.append(total_episode_reward)
    episode_lengths.append(steps_in_episode)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_episode_reward}, Epsilon: {epsilon:.2f}")

env.close()

# 4. Evaluate the learned policy (optional)
# You can run a few episodes with epsilon=0 to see the agent's performance
print("\n--- Evaluation ---")
env = gym.make('CliffWalking-v0', render_mode='human', is_slippery=False)  # 'human' for text-based display; other modes like 'rgb_array' available
test_episodes = 10
for _ in range(test_episodes):
    state, info = env.reset()
    done = False
    path = []
    while not done:
        path.append(state)
        action = np.argmax(Q[state, :]) # Pure exploitation
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
    path.append(state)
    print(f"Path taken: {path}")

# You can also visualize the learned policy on the grid
# (This would require custom rendering logic as env.render() is often limited)