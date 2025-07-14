# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

# Action Space
# The action shape is (1,) in the range {0, 3} indicating which direction to move the player.
# 0: Move left
# 1: Move down
# 2: Move right
# 3: Move up

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# "4x4": [
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFG"
# ]
#
# "8x8": [
#     "SFFFFFFF",
#     "FFFFFFFF",
#     "FFFHFFFF",
#     "FFFFFHFF",
#     "FFFHFFFF",
#     "FHHFFFHF",
#     "FHFFHFHF",
#     "FFFHFFFG",
# ]
# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
# env = gym.make('FrozenLake-v1', map_name="4x4", render_mode="human",)
env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human",)

obs, info = env.reset(seed=42)
done, total_reward, steps = False, 0, 0

while not done:
    action = env.action_space.sample()  # Random action (0: left, 1: down, 2: right, 3: up)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    print(obs, reward, terminated, truncated, info, done)

print(f"Total reward = {total_reward}, Episode length = {steps} steps")
env.close()