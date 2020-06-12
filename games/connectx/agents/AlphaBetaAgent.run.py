from kaggle_environments import make

from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent



env = make("connectx", debug=True)
env.render()
env.reset()
env.configuration.timeout = 24*60*60

action = AlphaBetaAgent.agent(env.state[0].observation, env.configuration)
print('action', action)
assert action == 3  # always play the middle square first

# env.run([AlphaBetaAgent.agent, "random"])
# # env.render(mode="ipython", width=500, height=450)
env.render(mode="human")
