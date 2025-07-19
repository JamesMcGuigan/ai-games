# https://gymnasium.farama.org/environments/toy_text/blackjack/
# Action Space       Discrete(2)
# Observation Space  Tuple(Discrete(32), Discrete(11), Discrete(2))
# Observation Values Player current sum, Dealer card, Usable Ace Boolean

# In Dealer.py we had a fixed policy of draw until 17 then stand
# MontyCarlo simulation wishes to test all possible stand-until policies, conditional on dealer card and useable ace

import gymnasium as gym
from collections import defaultdict
from collections import OrderedDict

# Create the environment
env = gym.make('Blackjack-v1')  # 'human' for text-based display; other modes like 'rgb_array' available

def play_blackjack(draw_until: int):
    # Reset to start a new episode
    observation, info = env.reset()  # Seed for reproducibility; returns initial observation and info dict
    player_sum, dealer_card, useable_ace = observation
    # print("Initial observation:", observation)  # e.g., (12, 5, 1)

    # Take actions in a loop until done
    done = False
    while not done:
        if player_sum >= draw_until: action = 0  # stand
        else:                        action = 1  # hit

        observation, reward, terminated, truncated, info = env.step(action)
        player_sum, dealer_card, useable_ace = observation
        done = terminated or truncated  # Episode ends on win/loss/draw or bust

        # env.render()  # Displays current state (text printout like "Player Sum: 16, Dealer Card: 6, Usable Ace: True")
        # print("Action", action, "Observation:", observation, "Reward:", reward, "Done:", done)

    return reward, player_sum, dealer_card, useable_ace


def monty_carlo(iterations: int):
    rewards = defaultdict(lambda: defaultdict(int))
    counts  = defaultdict(lambda: defaultdict(int))
    scores  = defaultdict(lambda: defaultdict(int))
    for i in range(iterations):
        for draw_until in range(21):
            reward, player_sum, dealer_card, useable_ace = play_blackjack(draw_until)
            rewards[ (useable_ace, dealer_card) ][ draw_until ] += reward
            counts[  (useable_ace, dealer_card) ][ draw_until ] += 1
            scores[  (useable_ace, dealer_card) ][ draw_until ] \
                = rewards[(useable_ace, dealer_card)][draw_until] / counts[(useable_ace, dealer_card)][draw_until]

    return scores

def print_monty_carlo_policy(scores):
    for (useable_ace, dealer_card), value in sorted(scores.items()):
        print(f"Ace {useable_ace} vs Dealer {dealer_card} = Draw Until", sorted(value.items(), key=lambda x: -x[1])[0])

scores = monty_carlo(1_000_000)
print_monty_carlo_policy(scores)

### Monty Carlo Policy (100,000)
# Ace 0 vs Dealer 1 = Draw Until (17, -0.3959375473700167)
# Ace 0 vs Dealer 2 = Draw Until (14, -0.019562179785747556)
# Ace 0 vs Dealer 3 = Draw Until (15, 0.0035082367297132396)
# Ace 0 vs Dealer 4 = Draw Until (12, 0.046442065491183876)
# Ace 0 vs Dealer 5 = Draw Until (12, 0.06289402581807266)
# Ace 0 vs Dealer 6 = Draw Until (13, 0.1122259956776783)
# Ace 0 vs Dealer 7 = Draw Until (17, 0.005933837709538644)
# Ace 0 vs Dealer 8 = Draw Until (16, -0.07243707796193984)
# Ace 0 vs Dealer 9 = Draw Until (15, -0.16073564593301434)
# Ace 0 vs Dealer 10= Draw Until (16, -0.2775583073767378)
# Ace 1 vs Dealer 1 = Draw Until (20, 0.5036390101892285)
# Ace 1 vs Dealer 2 = Draw Until (20, 0.9068219633943427)
# Ace 1 vs Dealer 3 = Draw Until (20, 0.8805309734513275)
# Ace 1 vs Dealer 4 = Draw Until (20, 0.8724340175953079)
# Ace 1 vs Dealer 5 = Draw Until (20, 0.9066022544283414)
# Ace 1 vs Dealer 6 = Draw Until (20, 0.9070796460176991)
# Ace 1 vs Dealer 7 = Draw Until (20, 0.9248826291079812)
# Ace 1 vs Dealer 8 = Draw Until (20, 0.9400584795321637)
# Ace 1 vs Dealer 9 = Draw Until (20, 0.931782945736434)
# Ace 1 vs Dealer 10 = Draw Until (20, 0.7910798122065728)

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
# Ace 1 vs Dealer 10 = Draw Until (20, 0.7766329556687704)