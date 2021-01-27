import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neural_network.RpsLSTM import RpsLSTM

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    n_epochs  = 10_000
    n_rounds  = 100
    model     = RpsLSTM(hidden_size=16, num_layers=2)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)


    for epoch in range(n_epochs):
        score    = 0
        count    = 0
        action   = None
        opponent = None
        loss     = Variable(torch.zeros((1,), requires_grad=True)).to(model.device)

        model.reset()
        optimizer.zero_grad()
        # Run entire game before calling loss.backward(); else:
        # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling backward the first time.
        for n in range(n_rounds):
            action, probs = model(action=action, opponent=opponent)
            opponent  = n % 3  # sequential agent
            loss     += model.loss(probs, opponent) / n_rounds
            score    += model.reward(action, opponent)
            count    += 1

        accuracy = score / count
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f'epoch = {epoch:6d} | accuracy = {accuracy*100:3.0f}% | loss = {loss.detach().item()}')
            if accuracy >= 0.99: break
