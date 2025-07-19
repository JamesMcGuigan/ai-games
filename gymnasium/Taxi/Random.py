# DOCS: https://gymnasium.farama.org/environments/toy_text/taxi/
import gymnasium as gym
import numpy as np

# env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
env = gym.make("Taxi-v3", render_mode="human")  # remove render_mode for speed
obs, info = env.reset(seed=42)
done, total_reward = False, 0

while not done:
    action = env.action_space.sample()              # 0 or 1 uniformly
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward, terminated, truncated, info, done)
    total_reward += reward

print(f"Episode length = {total_reward} time-steps")
env.close()