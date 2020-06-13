from kaggle_environments import make

from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent



env = make("connectx", debug=True)
env.render()
env.reset()

# env.run([AlphaBetaAgent.agent, "random"])
env.run([AlphaBetaAgent.agent, AlphaBetaAgent.agent])
# noinspection PyTypeChecker
env.render(mode="human")
