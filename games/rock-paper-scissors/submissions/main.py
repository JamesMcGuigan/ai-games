import os
current_path = os.path.abspath('..')

from simulations.Simulations import Simulations

simulations_instance = Simulations()
def simulations_agent(obs, conf):
    return simulations_instance.agent(obs, conf)
