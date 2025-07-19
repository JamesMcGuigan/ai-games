# https://gymnasium.farama.org/environments/toy_text/blackjack/
# Action Space       Discrete(2)
# Observation Space  Tuple(Discrete(32), Discrete(11), Discrete(2))
# Observation Values Player current sum, Dealer card, Usable Ace Boolean

import gymnasium as gym

# Create the environment
env = gym.make('Blackjack-v1', render_mode='human')  # 'human' for text-based display; other modes like 'rgb_array' available

def monty_carlo_policy(useable_ace, dealer_card):
    ### Monty Carlo Policy (1,000,000)
    # Ace 0 vs Dealer 1 = Draw Until (17, -0.42005639548836093)
    # Ace 0 vs Dealer 2 = Draw Until (13, -0.03538247135336017)
    # Ace 0 vs Dealer 3 = Draw Until (13, 0.0015871792896216966)
    # Ace 0 vs Dealer 4 = Draw Until (12, 0.03109689499141832)
    # Ace 0 vs Dealer 5 = Draw Until (13, 0.06090616489230007)
    # Ace 0 vs Dealer 6 = Draw Until (12, 0.09500210388517462)
    # Ace 0 vs Dealer 7 = Draw Until (17, 0.013589524366647168)
    # Ace 0 vs Dealer 8 = Draw Until (16, -0.07473711521145247)
    # Ace 0 vs Dealer 9 = Draw Until (17, -0.16213547199736658)
    # Ace 0 vs Dealer 10= Draw Until (16, -0.28115637597331206)
    # Ace 1 vs Dealer 1 = Draw Until (20, 0.5364167178856791)
    # Ace 1 vs Dealer 2 = Draw Until (20, 0.8777625361987502)
    # Ace 1 vs Dealer 3 = Draw Until (20, 0.8864440683182028)
    # Ace 1 vs Dealer 4 = Draw Until (20, 0.8901830282861897)
    # Ace 1 vs Dealer 5 = Draw Until (20, 0.8955359854236259)
    # Ace 1 vs Dealer 6 = Draw Until (20, 0.9044427123928293)
    # Ace 1 vs Dealer 7 = Draw Until (20, 0.9261632341723874)
    # Ace 1 vs Dealer 8 = Draw Until (20, 0.9291793313069909)
    # Ace 1 vs Dealer 9 = Draw Until (20, 0.9267649340574089)
    # Ace 1 vs Dealer 10= Draw Until (20, 0.7766329556687704)
    if   useable_ace:       draw_until = 20
    elif dealer_card == 1:  draw_until = 17
    elif dealer_card == 2:  draw_until = 13
    elif dealer_card == 3:  draw_until = 13
    elif dealer_card == 4:  draw_until = 12
    elif dealer_card == 5:  draw_until = 13
    elif dealer_card == 6:  draw_until = 12
    elif dealer_card == 7:  draw_until = 17
    elif dealer_card == 8:  draw_until = 16
    elif dealer_card == 9:  draw_until = 17
    elif dealer_card == 10: draw_until = 16
    else:                   draw_until = 17  # dealer default
    return draw_until


# Reset to start a new episode
observation, info = env.reset()  # Seed for reproducibility; returns initial observation and info dict
player_sum, dealer_card, useable_ace = observation
print("Initial observation:", observation)  # e.g., (12, 5, 1)

def play_blackjack():
    # Reset to start a new episode
    observation, info = env.reset()  # Seed for reproducibility; returns initial observation and info dict
    player_sum, dealer_card, useable_ace = observation
    print("Initial observation:", observation)  # e.g., (12, 5, 1)

    # Take actions in a loop until done
    done = False
    while not done:
        draw_until = monty_carlo_policy(useable_ace, dealer_card)
        if player_sum >= draw_until: action = 0  # stand
        else:                        action = 1  # hit

        observation, reward, terminated, truncated, info = env.step(action)
        player_sum, dealer_card, useable_ace = observation
        done = terminated or truncated  # Episode ends on win/loss/draw or bust

        env.render()  # Displays current state (text printout like "Player Sum: 16, Dealer Card: 6, Usable Ace: True")
        print("Action", action, "Observation:", observation, "Reward:", reward, "Done:", done)

    return reward, player_sum, dealer_card, useable_ace



total_rewards = 0
iterations = 100
for i in range(iterations):
    reward, player_sum, dealer_card, useable_ace = play_blackjack()
    total_rewards += reward
score = total_rewards / iterations
print(f"\nFinal Score: {total_rewards}/{iterations} = {score}")  # score: 12.0/100 = 0.12