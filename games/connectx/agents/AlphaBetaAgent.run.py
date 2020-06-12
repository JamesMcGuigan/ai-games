from kaggle_environments import make

from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent



env = make("connectx", debug=True)
env.render()
env.reset()

env.run([AlphaBetaAgent.agent, "random"])
# # env.render(mode="ipython", width=500, height=450)
env.render(mode="human")
