def sequential_agent(observation, configuration):
    action = observation.step % configuration.signs
    print(f'sequential_agent() = {action}')
    return int(action)
