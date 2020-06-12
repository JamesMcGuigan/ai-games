from kaggle_environments import make

from games.connectx.agents.old.RulesAgent import RulesAgent


#
# env = make("connectx", Environment(configuration={
#     # "episodeSteps": 0,
#     "agentTimeout": 360,
#     "actTimeout":   360,
#     "runTimeout":   360,
#     "rows":    4,
#     "columns": 7,
#     "inarow":  4,
#     "agentExec": "LOCAL"
# }))

env = make("connectx", debug=True)
env.render()
env.reset()
# Play as the first agent against default "random" agent.
env.run([RulesAgent, "random"])
# env.run([my_agent, my_agent])
# env.render(mode="ipython", width=500//2, height=450//2)
env.render(mode="human", width=500, height=450)
