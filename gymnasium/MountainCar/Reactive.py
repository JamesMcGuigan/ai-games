# CartPole-v1: A classic control problem where the goal is to balance a pole on a cart.
# HighScore = 121 timesteps to reach flag

# DOCS: https://gymnasium.farama.org/environments/classic_control/cart_pole/
# ChatGPT: https://chatgpt.com/c/6866ca65-6c94-8004-b19f-0ca449984947

### OBS
# Num, Observation, Min, Max
# 0, position of the car along the x-axis, -1.2,  0.6,  position (m)
# 1, velocity of the car,                  -0.07, 0.07, velocity (v)

# There are 3 discrete deterministic actions:
# 0: Accelerate to the left
# 1: Don’t accelerate
# 2: Accelerate to the right

import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")  # remove render_mode for speed
obs, info = env.reset(seed=42)
done, total_reward = False, 0

while not done:
    pos, velocity = obs

    action = 2                                     # start driving right uphill
    if   pos >= -0.5 and velocity > 0: action = 2  # continue right until velocity drags downhill
    elif pos >= -0.5 and velocity < 0: action = 0  # accelerate downhill to the left
    elif pos <= -0.5 and velocity < 0: action = 0  # accelerate uphill to the left
    elif pos <= -0.5 and velocity > 0: action = 2  # accelerate downhill to the right
    elif pos >= -1.0:                  action = 2  # Don't accelerate past left edge of board
    elif pos >= 0.5 and velocity == 0: action = 1  # STOP
    elif pos >= 0.5 and velocity > 0:  action = 2  # decelerate to stop down after flag
    else: action = 2                               # if in doubt accelerate right towards flag

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    print(action, obs, reward, terminated, truncated, info, done)

print(f"Episode length = {total_reward} time-steps")
env.close()