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
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
map_name="8x8"     # "4x4" | "8x8"
is_slippery=False
# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")


try:
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    # model = DQN.load("StableBaselines", env=env)
    model = PPO.load(f"StableBaselines-{map_name}-{is_slippery}", env=env)
except FileNotFoundError:
    # Instantiate the agent
    # model = DQN("MlpPolicy", env, verbose=1)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    # Save the agent
    model.save(f"StableBaselines-{map_name}-{is_slippery}")
    # del model  # delete trained model to demonstrate loading


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
