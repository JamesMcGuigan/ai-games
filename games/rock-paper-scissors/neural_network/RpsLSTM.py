import torch
import torch.nn as nn


# DOCS: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# DOCS: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# DOCS: https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/10-LSTM.ipynb
from neural_network.NNBase import NNBase


class RpsLSTM(NNBase):
# class RpsLSTM(nn.Module):

    def __init__(self, hidden_size=128, num_layers=3, dropout=0.25):
        super().__init__()
        self.device      = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.batch_size  = 1
        self.dropout     = dropout
        self.input_size  = self.cast_inputs(0,0).shape[-1]

        self.lstm  = nn.LSTM(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
            batch_first = True,
        )
        self.dense = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=2)
        self.reset()  # call before self.cast_inputs()
        self.to(self.device)


    def reset(self):
        self.stats  = torch.zeros((2,3), dtype=torch.float).to(self.device)
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )

    @staticmethod
    def reward(action: int, opponent: int) -> float:
        if (action - 1) % 3 == opponent % 3: return  1.0  # win
        if (action - 0) % 3 == opponent % 3: return  0.5  # draw
        if (action + 1) % 3 == opponent % 3: return  0.0  # loss
        return 0.0


    def loss(self, probs: torch.Tensor, opponent: int) -> torch.Tensor:
        ev = torch.zeros((3,), dtype=torch.float).to(self.device)
        ev[(opponent + 0) % 3] = 1.0   # expect rock, play paper + opponent rock     = win
        ev[(opponent + 1) % 3] = 0.5   # expect rock, play paper + opponent paper    = draw
        ev[(opponent + 2) % 3] = 0.0   # expect rock, play paper + opponent scissors = loss
        loss = -torch.sum(probs * (ev-1))
        return loss


    @staticmethod
    def cast_action(probs: torch.Tensor) -> int:
        expected = torch.argmax(probs, dim=2).detach().item()
        action   = int(expected + 1) % 3
        return action


    def cast_inputs(self, action: int, opponent: int) -> torch.Tensor:
        if not hasattr(self, 'stats'): self.reset()

        x = torch.zeros((2,3), dtype=torch.float).to(self.device)
        if action is not None:
            x[0, (action % 3)]   = 1.0
        if opponent is not None:
            x[1, (opponent % 3)] = 1.0

        # stats percentage frequency
        step  = torch.sum(self.stats[0])
        stats = ( self.stats / torch.sum(self.stats[0])
                  if step.item() > 0
                  else self.stats )
        x = torch.cat([x, stats], dim=0)
        # x = torch.cat([x.flatten(), step.reshape(1)])
        x = torch.reshape(x, (1,1,-1))    # (seq_len, batch, input_size)
        return x


    def forward(self, action: int, opponent: int):
        if action:   self.stats[0][action]   += 1.0
        if opponent: self.stats[1][opponent] += 1.0

        moves                 = self.cast_inputs(action, opponent)
        lstm_out, self.hidden = self.lstm(moves)
        lstm_out              = nn.functional.relu(lstm_out)
        expected_probs        = self.dense(lstm_out)
        expected_probs        = self.softmax(expected_probs)
        action                = self.cast_action(expected_probs)
        return action, expected_probs
