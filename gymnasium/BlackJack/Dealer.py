# https://gymnasium.farama.org/environments/toy_text/blackjack/
# Action Space       Discrete(2)
# Observation Space  Tuple(Discrete(32), Discrete(11), Discrete(2))
# Observation Values Player current sum, Dealer card, Usable Ace Boolean

import gymnasium as gym

# Create the environment
env = gym.make('Blackjack-v1', render_mode='human')  # 'human' for text-based display; other modes like 'rgb_array' available

# Reset to start a new episode
observation, info = env.reset()  # Seed for reproducibility; returns initial observation and info dict
player_sum, dealer_card, useable_ace = observation
print("Initial observation:", observation)  # e.g., (12, 5, 1)

# Take actions in a loop until done
done = False
while not done:
    # Player Policy same as dealer
    action = 0
    if player_sum >= 17: action = 0  # stand
    if player_sum <= 16: action = 1  # hit

    observation, reward, terminated, truncated, info = env.step(action)
    player_sum, dealer_card, useable_ace = observation
    done = terminated or truncated  # Episode ends on win/loss/draw or bust
    env.render()  # Displays current state (text printout like "Player Sum: 16, Dealer Card: 6, Usable Ace: True")
    print("Action", action, "Observation:", observation, "Reward:", reward, "Done:", done)

# Clean up
# sleep(10)
# env.close()