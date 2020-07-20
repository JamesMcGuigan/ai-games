from kaggle_environments import make

from agents.AlphaBetaAgent import AlphaBetaAgent



env = make("connectx", debug=True)
env.render()
env.reset()

# env.run([AlphaBetaAgent.agent, "random"])
env.run([ AlphaBetaAgent.agent(search_max_depth=3), AlphaBetaAgent.agent(search_max_depth=3) ])
# noinspection PyTypeChecker
env.render(mode="human")

# Timings
