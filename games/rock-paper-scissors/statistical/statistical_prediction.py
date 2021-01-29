# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-statistical-prediction

import random
from collections import Counter

# Create a small amount of starting history
stat_pred_history = {
    "guess":      [0,1,2],
    "prediction": [0,1,2],
    "expected":   [0,1,2],
    "action":     [1,2,0],
    "opponent":   [0,1],
    "rotn":       [0,1],
}

# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 1000, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
def statistical_prediction_agent(observation, configuration, verbose=True):
    global stat_pred_history
    actions          = list(range(configuration.signs))  # [0,1,2]
    last_action      = stat_pred_history['action'][-1]
    prev_opp_action  = stat_pred_history['opponent'][-1]
    opponent_action  = observation.lastOpponentAction if observation.step > 0 else 2
    rotn             = (opponent_action - prev_opp_action) % configuration.signs

    stat_pred_history['opponent'].append(opponent_action)
    stat_pred_history['rotn'].append(rotn)

    # Make weighted random guess based on the complete move stat_pred_history, weighted towards relative moves based on our last action
    move_frequency   = Counter(stat_pred_history['rotn'])
    action_frequency = Counter(zip(stat_pred_history['action'], stat_pred_history['rotn']))
    move_weights     = [ move_frequency.get(n, 1)
                         + action_frequency.get((last_action,n), 1)
                         for n in range(configuration.signs) ]
    guess            = random.choices( population=actions, weights=move_weights, k=1 )[0]

    # Compare our guess to how our opponent actually played
    guess_frequency  = Counter(zip(stat_pred_history['guess'], stat_pred_history['rotn']))
    guess_weights    = [ guess_frequency.get((guess,n), 1)
                         for n in range(configuration.signs) ]
    prediction       = random.choices( population=actions, weights=guess_weights, k=1 )[0]

    # Repeat, but based on how many times our prediction was correct
    pred_frequency   = Counter(zip(stat_pred_history['prediction'], stat_pred_history['rotn']))
    pred_weights     = [ pred_frequency.get((prediction,n), 1)
                         for n in range(configuration.signs) ]
    expected         = random.choices( population=actions, weights=pred_weights, k=1 )[0]


    # Slowly decay to 50% pure randomness as the match progresses
    pure_random_chance = observation.step / (configuration.episodeSteps * 2)
    if random.random() < pure_random_chance:
        action = random.randint(0, configuration.signs-1)
        is_pure_random_chance = True
    else:
        # Play the +1 counter move
        # action = (expected + 1) % configuration.signs                  # without rotn
        action = (opponent_action + expected + 1) % configuration.signs  # using   rotn
        is_pure_random_chance = False

    # Persist state
    stat_pred_history['guess'].append(guess)
    stat_pred_history['prediction'].append(prediction)
    stat_pred_history['expected'].append(expected)
    stat_pred_history['action'].append(action)

    # Print debug information
    if verbose:
        print('step                      = ', observation.step)
        print('opponent_action           = ', opponent_action)
        print('guess,      move_weights  = ', guess,      move_weights)
        print('prediction, guess_weights = ', prediction, guess_weights)
        print('expected,   pred_weights  = ', expected,   pred_weights)
        print('action                    = ', action)
        print('pure_random_chance        = ', f'{100*pure_random_chance:.2f}%', is_pure_random_chance)
        print()

    return action
