# SOLVED by Grok: https://grok.com/chat/91aceab9-0be9-4165-ba4b-697f4a642787
#
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
done, total_reward, steps = False, 0, 0

while not done:
    pos_x, pos_y, vel_x, vel_y, angle, angular_velocity, left_leg, right_leg = obs
    angle_targ = pos_x * 0.5 + vel_x * 1.0
    if angle_targ > 0.4: angle_targ = 0.4
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(pos_x)
    angle_todo = (angle_targ - angle) * 0.5 - angular_velocity * 1.0
    hover_todo = (hover_targ - pos_y) * 0.5 - vel_y * 0.5
    if left_leg or right_leg:
        angle_todo = 0
        hover_todo = -vel_y * 0.5
    if   hover_todo > np.abs(angle_todo) and hover_todo > 0.05: action = 2
    elif angle_todo < -0.05:                                    action = 3
    elif angle_todo > +0.05:                                    action = 1
    else:                                                       action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    print(obs, reward, terminated, truncated, info, done)

print(f"Total reward = {total_reward}, Episode length = {steps} steps")
env.close()