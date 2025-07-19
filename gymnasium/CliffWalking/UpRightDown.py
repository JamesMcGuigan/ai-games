# https://gymnasium.farama.org/environments/toy_text/cliff_walking/
# Action Space
# 0: Move up
# 1: Move right
# 2: Move down
# 3: Move left

# Observation Space
# The agent moves through a  4×12 gridworld, with states numbered as follows:
# At the start of any episode, state 36 is the initial state.
# State 47 is the only terminal state, and the cliff corresponds to states 37 through 46.
# [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
#  [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
#  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]

import gymnasium as gym

# Create the environment
env = gym.make('CliffWalking-v0', render_mode='human', is_slippery=True)  # 'human' for text-based display; other modes like 'rgb_array' available

# Reset to start a new episode
observation, info = env.reset(seed=42)  # Seed for reproducibility; returns initial observation and info dict
print("Initial observation:", observation)  # e.g., (12, 5, 1)

# Take actions in a loop until done
done = False
while not done:
    if   observation in [12,24,36]:                action = 0  # UP
    elif observation in [11,23,35,47]:             action = 2  # DOWN
    elif observation in [0,1,2,3,4,5,6,7,8,9,10]:  action = 1  # RIGHT
    else:                                          action = 0  # UP
    print(observation, env.action_space)


    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # Episode ends on win/loss/draw or bust
    env.render()  # Displays current state (text printout like "Player Sum: 16, Dealer Card: 6, Usable Ace: True")
    # print("Observation:", observation, "Reward:", reward, "Done:", done)

# Clean up
env.close()