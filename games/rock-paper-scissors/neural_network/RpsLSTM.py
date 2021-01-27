# DOCS: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# DOCS: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# DOCS: https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/10-LSTM.ipynb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class RpsLSTM(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, dropout=0.25, input_size=6):
        super().__init__()
        self.device      = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.batch_size  = 1

        self.lstm  = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=2)
        self.reset()

    def reset(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )


    @staticmethod
    def cast_inputs(action: int, opponent: int) -> torch.Tensor:
        x = torch.zeros((3,2), dtype=torch.float)
        if action is not None:
            x[action % 3, 0] = 1.0
        if opponent is not None:
            x[opponent % 3, 0] = 1.0
        x = torch.reshape(x, (1,1,6))  # (seq_len, batch, input_size)
        return x.to(device)

    @staticmethod
    def loss(probs: torch.Tensor, opponent: int) -> torch.Tensor:
        ev = torch.zeros((3,), dtype=torch.float).to(device)
        ev[(opponent + 0) % 3] = 1.0   # expect rock, play paper + opponent rock     = win
        ev[(opponent + 1) % 3] = 0.75  # expect rock, play paper + opponent paper    = draw
        ev[(opponent + 2) % 3] = 0.0   # expect rock, play paper + opponent scissors = loss
        loss = -torch.sum(probs * (ev-1))
        return loss


    @staticmethod
    def cast_action(probs: torch.Tensor) -> int:
        expected = torch.argmax(probs, dim=2).detach().item()
        action   = int(expected + 1) % 3
        return action


    def forward(self, action: int, opponent: int):
        moves                 = self.cast_inputs(action, opponent)
        lstm_out, self.hidden = self.lstm(moves)
        lstm_out              = nn.functional.relu(lstm_out)
        expected_probs        = self.dense(lstm_out)
        expected_probs        = self.softmax(expected_probs)
        action                = model.cast_action(expected_probs)
        return action, expected_probs


def get_reward(action: int, opponent: int) -> float:
    if action == (opponent + 1) % 3: return  1.0  # win
    if action == (opponent + 0) % 3: return  0.5  # draw
    if action == (opponent - 1) % 3: return  0.0  # loss
    return 0.0



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    n_rounds  = 100
    model     = RpsLSTM(hidden_size=16, num_layers=2).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)


    for epoch in range(1_000):
        score    = 0
        count    = 0
        action   = None
        opponent = None
        loss     = Variable(torch.zeros((1,), requires_grad=True)).to(device)

        model.reset()
        optimizer.zero_grad()
        # Run entire game before calling loss.backward(); else:
        # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling backward the first time.
        for n in range(n_rounds):
            action, probs = model(action=action, opponent=opponent)
            opponent  = n % 3  # sequential agent
            loss     += model.loss(probs, opponent) / n_rounds
            score    += get_reward(action, opponent)
            count    += 1

        accuracy = score / count
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f'{epoch:6d} | {accuracy*100:3.0f}% | loss = {loss.detach().item()}')
            if score == count: break
