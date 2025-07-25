# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading

import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")


try:
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    # model = DQN.load("StableBaselines", env=env)
    model = PPO.load("StableBaselines", env=env)
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
    model.save("StableBaselines")
    # del model  # delete trained model to demonstrate loading


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
