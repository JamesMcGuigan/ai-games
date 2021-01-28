import random
import sys
from typing import Dict

import torch
from kaggle_environments import make
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from neural_network.RpsLSTM import RpsLSTM
from roshambo_competition.anti_rotn import anti_rotn
from simple.paper import paper_agent
from simple.rock import rock_agent
from simple.scissors import scissors_agent
from simple.sequential import sequential_agent


def rps_trainer(model, agents: Dict, steps=100, max_epochs=10_000, lr=1e-3, log_freq=10):
    try:
        env   = make("rps", { "episodeSteps": steps }, debug=False)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100)

        for epoch in range(max_epochs):
            scores = Variable(torch.zeros((len(agents),), requires_grad=True)).to(model.device)
            losses = Variable(torch.zeros((len(agents),), requires_grad=True)).to(model.device)

            for agent_id, (agent_name, agent) in enumerate(agents.items()):
                trainer  = env.train(random.sample([None, agent], 2))  # random player order
                observation = trainer.reset()

                model.reset()
                optimizer.zero_grad()

                action   = None
                opponent = None
                for step in range(1,sys.maxsize):
                    action, probs = model.forward(action=action, opponent=opponent)

                    observation, reward, done, info = trainer.step(action)
                    opponent = observation.lastOpponentAction

                    losses[agent_id] += model.loss(probs, opponent)
                    scores[agent_id] += (reward + 1.0) / 2.0
                    if done:
                        scores[agent_id] /= step
                        losses[agent_id] /= step
                        break

            # print(env.render(mode='ansi'))
            loss = -torch.sum(torch.log(1-losses))  # cross entropy loss
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step(loss)

            if epoch % log_freq == 0:
                accuracy = " ".join([
                    f'{(scores[n] * 100).int().item():3d} {name}'
                    for n, name in enumerate(agents.keys())
                ])
                print(f'{epoch:6d} | loss = {loss.detach().item():.8f} | {accuracy}')
                if torch.mean(scores).item() >= (1 - 2/steps): break  # allowed first 2 moves wrong

    except KeyboardInterrupt: pass
    model.save()


if __name__ == '__main__':
    model  = RpsLSTM(hidden_size=128, num_layers=3, dropout=0.25).train()
    agents = {
        'r': rock_agent,
        'p': paper_agent,
        's': scissors_agent,
        # 'pi': pi_agent,
        # '-pi': anti_pi_agent,
        'seq':  sequential_agent,
        'rotn': anti_rotn
    }
    rps_trainer(model, agents, steps=100, lr=1e-4)
