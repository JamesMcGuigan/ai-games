# White Belt Agents from https://www.kaggle.com/chankhavu/rps-dojo

statistical_action_histogram = {}
def statistical(observation, configuration):
    global statistical_action_histogram
    if observation.step == 0:
        statistical_action_histogram = {}
        return
    action = observation.lastOpponentAction
    if action not in statistical_action_histogram:
        statistical_action_histogram[action] = 0
    statistical_action_histogram[action] += 1
    mode_action = None
    mode_action_count = None
    for k, v in statistical_action_histogram.items():
        if mode_action_count is None or v > mode_action_count:
            mode_action = k
            mode_action_count = v
            continue

    return (mode_action + 1) % configuration.signs
