# %%writefile anti_rotn.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-rotn

import random

anti_rotn_history = []
anti_rotn_stats   = [0, 0, 0]

# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def anti_rotn(observation, configuration, warmup=1):
    global anti_rotn_history
    global anti_rotn_stats

    if observation.step > 0:
        anti_rotn_history.append(observation.lastOpponentAction)

    if len(anti_rotn_history) >= 2:
        rotn = (anti_rotn_history[-1] - anti_rotn_history[-2]) % configuration.signs
        anti_rotn_stats[ rotn] += 1

    if observation.step < warmup:
        action = random.randint(0, configuration.signs-1)
    else:
        ev = list({
            0: anti_rotn_stats[2] - anti_rotn_stats[1],
            1: anti_rotn_stats[0] - anti_rotn_stats[2],
            2: anti_rotn_stats[1] - anti_rotn_stats[0],
        }.values())
        offset = ev.index(max(ev))
        action = (offset + observation.lastOpponentAction) % configuration.signs
        print(f'rotn_stats = {anti_rotn_stats} | ev = {ev} | action = {action}')

    return int(action)
