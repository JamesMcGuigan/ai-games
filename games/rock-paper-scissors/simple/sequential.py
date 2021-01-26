def sequential_agent(observation, configuration):
    return observation.step % configuration.signs
