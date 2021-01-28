import random
import sys
from typing import Dict

import torch
from kaggle_environments import make
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from neural_network.RpsLSTM import RpsLSTM
from roshambo_competition.anti_rotn import anti_rotn
from roshambo_competition.greenberg import greenberg_agent
from roshambo_competition.iocaine_powder import iocaine_agent
from simple.paper import paper_agent
from simple.rock import rock_agent
from simple.scissors import scissors_agent
from simple.sequential import sequential_agent
from statistical.statistical_prediction import statistical_prediction_agent
from tree.multi_stage_decision_tree import decision_tree_agent


def rps_trainer(model, agents: Dict, steps=100, max_epochs=10_000, lr=1e-3, log_freq=10):
    try:
        env   = make("rps", { "episodeSteps": steps }, debug=False)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        scheduler = None
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100)

        accuracies     = { agent_name: 0.0 for agent_name in agents.keys()}
        running_losses = torch.zeros((1,)).to(model.device)
        for epoch in range(max_epochs):
            # skip high-accuracy agents more often, but train at least 10% of the time
            selected_agents = {
                agent_name: agent
                for (agent_name, agent) in agents.items()
                if random.random() + 0.1 >= accuracies[agent_name]
            }
            if len(selected_agents) == 0: continue

            scores = Variable(torch.zeros((   len(selected_agents),), requires_grad=True)).to(model.device)
            losses = Variable(torch.zeros((2, len(selected_agents),), requires_grad=True)).to(model.device)

            for agent_index, (agent_name, agent) in enumerate(selected_agents.items()):
                trainer     = env.train(random.sample([None, agent], 2))  # random player order
                observation = trainer.reset()

                model.reset()
                optimizer.zero_grad()

                action     = None
                opponent   = None
                for step in range(1,sys.maxsize):
                    action, probs, hash_id = model.forward(action=action, opponent=opponent)

                    observation, reward, done, info = trainer.step(action)
                    opponent = observation.lastOpponentAction

                    losses[0][agent_index] += model.loss_probs(probs, opponent)
                    losses[1][agent_index] += model.loss_hash(hash_id, agent_name)
                    scores[agent_index]    += (reward + 1.0) / 2.0
                    if done: break

                losses[0][agent_index] /= step  # NOTE: steps = 2 * step
                losses[1][agent_index] /= step
                scores[agent_index]    /= step
                accuracies[agent_name]  = ( (accuracies[agent_name] + scores[agent_index].item())
                                            / (2 if sum(accuracies.values()) else 1) )

            # print(env.render(mode='ansi'))
            running_losses = ( (running_losses + torch.mean(losses, dim=1))
                               / (2 if torch.sum(running_losses) else 1) )
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step(loss)

            if epoch % log_freq == 0:
                accuracy_log = " ".join([
                    f'{round(value * 100):3d} {name}'
                    for name, value in accuracies.items()
                ])
                print(f'{epoch:6d} | losses = {running_losses[0].item():.6f} {running_losses[1].item():.6f} | {accuracy_log}')
                if torch.mean(scores).item() >= (1 - 2/steps): break  # allowed first 2 moves wrong

    except KeyboardInterrupt: pass


if __name__ == '__main__':
    agents = {
        'r':         rock_agent,
        'p':         paper_agent,
        's':         scissors_agent,
        'seq':       sequential_agent,
        'rotn':      anti_rotn,
        'tree':      decision_tree_agent,
        'iocaine':   iocaine_agent,
        'greenberg': greenberg_agent,
        'stats':     statistical_prediction_agent,
    }

    model = RpsLSTM(hidden_size=128, num_layers=3, dropout=0.25).train()
    model.load()
    rps_trainer(model, agents, steps=100, lr=1e-4)
    model.save()
