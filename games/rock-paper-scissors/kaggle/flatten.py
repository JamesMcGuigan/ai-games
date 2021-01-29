# %writefile flatten.py
# Source: https://www.kaggle.com/tonyrobinson/flatten
import math
import random
import numpy as np


class FlattenAgent:
    pFlatten = 0.7 # prob of playing Flatten, else do simple predict
    offset   = 2.0
    halfLife = 100.0
    countPow = math.exp(math.log(2)/halfLife)

    naction = 3
    reward  = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


    def __init__(self):
        self.countInc = 1e-30
        self.countOp  = self.countInc * np.ones((self.naction, self.naction, self.naction))
        self.countAg  = self.countInc * np.ones((self.naction, self.naction, self.naction))
        self.histAgent = []    # Agent history
        self.histOpponent = [] # Opponent history
        self.nwin = 0


    # make a move
    def move(self, lastOpponentAction, step):
        if step == 0:
            dist = np.ones(self.naction)
        else:
            # store last opponent action
            self.histOpponent.append(lastOpponentAction)

            # score last game
            self.nwin += self.reward[self.histAgent[-1], lastOpponentAction]
            print('step: ', step, 'win: ', self.nwin)

            if step > 1:
                # increment predictors
                self.countOp[self.histOpponent[-2], self.histAgent[-2], self.histOpponent[-1]] += self.countInc
                self.countAg[self.histOpponent[-2], self.histAgent[-2], self.histAgent[-1]] += self.countInc

            # decide on what strategy to play
            if len(self.histOpponent) < 2:
                dist = np.ones(self.naction)
            else:
                if random.random() < self.pFlatten:
                    # stochastically flatten the distribution
                    count = self.countAg[self.histOpponent[-1], self.histAgent[-1]]
                    dist  = (self.offset + 1) * count.max() - self.offset * count.min() - count
                else:
                    # simple prediction of opponent
                    count = self.countOp[self.histOpponent[-1], self.histAgent[-1]]
                    gain  = np.dot(self.reward, count)
                    dist  = gain + gain.min()

        agentAction = random.choices(range(self.naction), weights=dist)[0]
        self.histAgent.append(agentAction)
        self.countInc *= self.countPow

        return(agentAction)


flatten_instance = None
def flatten_agent(observation, configuration):
    global flatten_instance

    if flatten_instance == None:
        flatten_instance = FlattenAgent()
        return flatten_instance.move(None, observation.step)
    else:
        return flatten_instance.move(observation.lastOpponentAction, observation.step)
