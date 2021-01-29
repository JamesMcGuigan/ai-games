# White Belt Agents from https://www.kaggle.com/chankhavu/rps-dojo

def mirror_opponent_agent(observation, configuration):
    if observation.step > 0:
        return observation.lastOpponentAction
    else:
        return 0
