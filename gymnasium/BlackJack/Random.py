import gymnasium as gym

# Create the environment
env = gym.make('Blackjack-v1', render_mode='human')  # 'human' for text-based display; other modes like 'rgb_array' available

# Reset to start a new episode
observation, info = env.reset(seed=42)  # Seed for reproducibility; returns initial observation and info dict
print("Initial observation:", observation)  # e.g., (12, 5, 1)

# Take actions in a loop until done
done = False
while not done:
    action = env.action_space.sample()  # Random action (0 or 1); replace with your policy
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # Episode ends on win/loss/draw or bust
    env.render()  # Displays current state (text printout like "Player Sum: 16, Dealer Card: 6, Usable Ace: True")
    print("Observation:", observation, "Reward:", reward, "Done:", done)

# Clean up
env.close()