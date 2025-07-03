# CartPole-v1: A classic control problem where the goal is to balance a pole on a cart.
# HighScore = 500 steps using: action = (angle + angular_velocity/10 > 0)

# DOCS: https://gymnasium.farama.org/environments/classic_control/cart_pole/
# ChatGPT: https://chatgpt.com/c/6866ca65-6c94-8004-b19f-0ca449984947

### OBS
# Num, Observation, Min, Max
# 0, Cart Position, -4.8, 4.8
# 1, Cart Velocity, -Inf, Inf
# 2, Pole Angle, ~ -0.418 rad (-24°), ~ 0.418 rad (24°)
# 3, Pole Angular Velocity, -Inf, Inf

import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")  # remove render_mode for speed
obs, info = env.reset(seed=42)
done, total_reward = False, 0

while not done:
    pos, velocity, angle, angular_velocity = obs
    if   angle + angular_velocity/10 > 0: action = 1
    elif angle + angular_velocity/10 < 0: action = 0
    else:                              action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    print(obs, reward, terminated, truncated, info, done)

print(f"Episode length = {total_reward} time-steps")
env.close()