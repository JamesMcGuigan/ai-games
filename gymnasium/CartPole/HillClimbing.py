# https://learn.udacity.com/nanodegrees/nd893/parts/cd1763/lessons/22cf8ac3-5823-404c-b825-5dcda25a736a/concepts/2b715c20-cfb5-4fac-8a51-c55cfa3edd85?lesson_tab=lesson
# BUG: AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?

import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="rgb_array")
print('observation space:', env.observation_space)
print('action space:', env.action_space)


class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.w)  # same output as np.matmul
        # x = np.matmul(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action



def hill_climbing(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.inf
    best_w = policy.w
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state, _ = env.reset()
        for t in range(max_t):
            action = int(policy.act(state))
            state, reward, done, _, _ = env.step(action)  # AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            policy.w = best_w
            break

    return scores


policy = Policy()
scores = hill_climbing()


env = gym.make("CartPole-v1", render_mode="human")  # remove render_mode for speed
state, _ = env.reset()
for t in range(1000):
    action = policy.act(state)
    state, reward, done, _, _ = env.step(action)
    if done:
        break

env.close()