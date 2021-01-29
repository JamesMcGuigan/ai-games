# %%writefile anti_rotn.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-rotn-weighted-random

import random
import numpy as np

rotn_history    = []
rotn_stats = np.array([0, 0, 0], dtype=np.float)

# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def anti_rotn(observation, configuration, warmup=10, average='mean', min_weight=2, decay=0.95 ):
    assert average in ['running', 'mean']
    global rotn_history
    global rotn_stats

    if observation.step > 0:
        rotn_history.append( observation.lastOpponentAction )

    if len(rotn_history) >= 2:
        rotn = (rotn_history[-1] - rotn_history[-2]) % configuration.signs
        if average == 'running':
            rotn_stats[ rotn ] += observation.step
        else:
            rotn_stats[ rotn ] += 1
    rotn_stats *= decay

    if observation.step < warmup:
        action = random.randint(0, configuration.signs-1)
    else:
        ev = np.array(list({
            0: (rotn_stats[2] - rotn_stats[1]),
            1: (rotn_stats[0] - rotn_stats[2]),
            2: (rotn_stats[1] - rotn_stats[0]),
        }.values()))
        ev = ev + np.max(ev) * min_weight
        ev[ ev < 0 ] = 0
        if sum(ev): ev = (ev / sum(ev)).round(3)

        # offset = ev.index(max(ev)  # original anti-rotn spec
        offset = random.choices( population=[0,1,2], weights=ev, k=1 )[0]
        action = (offset + observation.lastOpponentAction) % configuration.signs
        print(f'rotn_stats = {rotn_stats.round(3).tolist()} | ev = {ev.tolist()} | action = {action}')

    return int(action)
