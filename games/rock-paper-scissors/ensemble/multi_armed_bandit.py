# %%writefile multi_armed_bandit.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-armed-bandit/
import contextlib
import os
from collections import defaultdict
import numpy as np
import time
from operator import itemgetter

from memory.memory_patterns import memory_patterns_agent
from memory.RPSNaiveBayes import naive_bayes_agent
from rng.random_agent import random_agent
from roshambo_competition.anti_rotn import anti_rotn
from roshambo_competition.greenberg import greenberg_agent
from roshambo_competition.iocaine_powder import iocaine_agent
from simple.anti_pi import anti_pi_agent
from statistical.statistical_prediction import statistical_prediction_agent
# from simple.anti_pi import anti_pi_agent
# from simple.pi import pi_agent

mlb_history  = {
    'actions':  [],
    'opponent': []
}
mlb_expected = defaultdict(list)
mlb_agents   = {
    #     'random':               (lambda obs, conf: random_agent(obs, conf)),
    #     'pi':                   (lambda obs, conf: pi_agent(obs, conf)),
    'anti_pi':               (lambda obs, conf: anti_pi_agent(obs, conf)),
    #     'anti_anti_pi':         (lambda obs, conf: anti_anti_pi_agent(obs, conf)),
    #     'reactionary':          (lambda obs, conf: reactionary(obs, conf)),
    'anti_rotn':            (lambda obs, conf: anti_rotn(obs, conf, warmup=1)),

    'iou2':                  (lambda obs, conf: iou2_agent(obs, conf)),
    'geometry':              (lambda obs, conf: geometry_agent(obs, conf)),
    'memory_patterns_v20':   (lambda obs, conf: memory_patterns_v20(obs, conf)),
    'testing_please_ignore': (lambda obs, conf: testing_please_ignore(obs, conf)),
    'bumblepuppy':           (lambda obs, conf: centrifugal_bumblepuppy(obs, conf)),
    'dllu1_agent':           (lambda obs, conf: dllu1_agent(obs, conf)),

    'genetics':              (lambda obs, conf: genetics_choice(obs, conf)),
    'flatten':               (lambda obs, conf: flatten_agent(obs, conf)),
    'transition':            (lambda obs, conf: transition_agent(obs, conf)),
    # 'kumoko':                (lambda obs, conf: kumoko_agent(obs, conf)), # broken    

    'memory_patterns':       (lambda obs, conf: memory_patterns(obs, conf)),
    'naive_bayes':           (lambda obs, conf: naive_bayes(obs, conf)),
    'iocaine':               (lambda obs, conf: iocaine_agent(obs, conf)),
    'greenberg':             (lambda obs, conf: greenberg_agent(obs, conf)),
    'statistical':           (lambda obs, conf: statistical_prediction_agent(obs, conf)),
    'statistical_expected':  (lambda obs, conf: statistical_history['expected'][-1] + 1),
    # 'decision_tree_1':       (lambda obs, conf: decision_tree_agent_1(obs, conf, stages=1, window=20)),
    'decision_tree_2':       (lambda obs, conf: decision_tree_agent_2(obs, conf, stages=2, window=6)),
    'decision_tree_3':       (lambda obs, conf: decision_tree_agent_3(obs, conf, stages=3, window=10)),
    #'random_seed_search':    (lambda obs, conf: random_seed_search_agent(obs, conf)),
}

# observation   = {'step': 1, 'lastOpponentAction': 1}
# configuration = {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def multi_armed_bandit_agent(observation, configuration, warmup=1, step_reward=3, decay_rate=0.95, verbose=True ):
    global mlb_expected
    global mlb_history
    global mlb_agents
    time_start = time.perf_counter()
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') == 'Interactive':
        warmup = 1


    if observation.step != 0:
        mlb_history['opponent'] += [ observation.lastOpponentAction ]
    # else:
    #     mlb_history['opponent'] += [ random_agent(observation, configuration) ]


    # Implement Multi Armed Bandit Logic
    win_loss_scores = defaultdict(lambda: [0.0, 0.0])
    for name, values in list(mlb_expected.items()):
        for n in range(min(len(values), len(mlb_history['opponent']))):
            win_loss_scores[name][1] = (win_loss_scores[name][1] - 1) * decay_rate + 1
            win_loss_scores[name][0] = (win_loss_scores[name][0] - 1) * decay_rate + 1

            # win | expect rock, play paper -> opponent plays rock
            if   mlb_expected[name][n] == (mlb_history['opponent'][n] + 0) % configuration.signs:
                win_loss_scores[name][0] += step_reward

                # draw | expect rock, play paper -> opponent plays paper
            elif mlb_expected[name][n] == (mlb_history['opponent'][n] + 1) % configuration.signs:
                win_loss_scores[name][0] += step_reward
                win_loss_scores[name][1] += step_reward

                # win | expect rock, play paper -> opponent plays scissors
            elif mlb_expected[name][n] == (mlb_history['opponent'][n] + 2) % configuration.signs:
                win_loss_scores[name][1] += step_reward


    # Update predictions for next turn
    for name, agent_fn in list(mlb_agents.items()):
        try:
            with contextlib.redirect_stdout(None):  # disable stdout for child agents
                agent_action        = agent_fn(observation, configuration)
                agent_expected      = (agent_action - 1) % configuration.signs
                mlb_expected[name] += [ agent_expected ]
        except Exception as exception:
            print('Exception:', name, agent_fn, exception)


    # Pick the Best Agent
    beta_scores = {
        name: np.random.beta(win_loss_scores[name][0], win_loss_scores[name][1])
        for name in win_loss_scores.keys()
    }

    if observation.step == 0:
        # Always play scissors first move
        # At Auction       - https://www.artsy.net/article/artsy-editorial-christies-sothebys-played-rock-paper-scissors-20-million-consignment
        # EDA best by test - https://www.kaggle.com/jamesmcguigan/rps-episode-archive-dataset-eda
        agent_name = 'scissors'
        expected = 1
    elif observation.step < warmup:
        agent_name = 'random'
        expected   = random_agent(observation, configuration)
    else:
        agent_name = sorted(beta_scores.items(), key=itemgetter(1), reverse=True)[0][0]
        expected   = mlb_expected[agent_name][-1]

    action = (expected + 1) % configuration.signs


    if verbose:
        best_score    = beta_scores.get(agent_name,0)
        last_opponent = (mlb_history['opponent'] or [0])[-1]
        win_symbol    = (
            ' ' if observation.step == 0 else
            '+' if mlb_history['actions'][-1] == (mlb_history['opponent'][-1] + 1) % 3 else
            '|' if mlb_history['actions'][-1] == (mlb_history['opponent'][-1] + 0) % 3 else
            '-'
        )
        time_taken    = time.perf_counter() - time_start
        print(f'{observation.step:4d} | {time_taken:0.2f}s | {last_opponent}{win_symbol} -> action = {expected} -> {action} | {best_score*100:3.0f}% {agent_name}')

    mlb_history['actions'] += [ action ]
    return int(action)
