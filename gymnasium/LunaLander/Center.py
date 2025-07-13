# https://gymnasium.farama.org/environments/box2d/lunar_lander/
# OBS
# Num, Observation, Min, Max
# 0, x-coordinate of the lander, -2.5, 2.5
# 1, y-coordinate of the lander, -2.5, 2.5
# 2, Linear velocity in x, -10, 10
# 3, Linear velocity in y, -10, 10
# 4, Angle, -6.2831855, 6.2831855
# 5, Angular velocity, -10, 10
# 6, Leg 1 contact, 0, 1
# 7, Leg 2 contact, 0, 1

# ACTION SPACE
# Discrete(4):
# 0: Do nothing
# 1: Fire left orientation engine
# 2: Fire main engine
# 3: Fire right orientation engine
# Source: gymnasium.farama.org

import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3",
                render_mode="human",
                continuous=False,
                enable_wind=True,
                gravity=-10.0,
                wind_power=15.0,
                turbulence_power=1.5
)
obs, info = env.reset(seed=42)
pos_x, pos_y, vel_x, vel_y, angle, angular_velocity, left_leg, right_leg = obs
done, total_reward, steps = False, 0, 0

while not done:
    action = 0                  # (0: nothing, 1: left, 2: main, 3: right)
    if pos_x > 0.0: action = 3  # (0: nothing, 1: left, 2: main, 3: right)
    if pos_x < 0.0: action = 1  # (0: nothing, 1: left, 2: main, 3: right)

    obs, reward, terminated, truncated, info = env.step(action)
    pos_x, pos_y, vel_x, vel_y, angle, angular_velocity, left_leg, right_leg = obs
    done = terminated or truncated
    total_reward += reward
    steps += 1
    print(obs, reward, terminated, truncated, info, done)

print(f"Total reward = {total_reward}, Episode length = {steps} steps")
env.close()