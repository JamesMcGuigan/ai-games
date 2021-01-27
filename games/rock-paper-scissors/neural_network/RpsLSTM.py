import torch
import torch.nn as nn


# DOCS: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# DOCS: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# DOCS: https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/10-LSTM.ipynb
from neural_network.NNBase import NNBase


class RpsLSTM(NNBase):
# class RpsLSTM(nn.Module):

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
        self.to(self.device)


    def reset(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )


    def cast_inputs(self, action: int, opponent: int) -> torch.Tensor:
        x = torch.zeros((2,3), dtype=torch.float).to(self.device)
        if action is not None:
            x[0, (action % 3)]   = 1.0
        if opponent is not None:
            x[1, (opponent % 3)] = 1.0
        x = torch.reshape(x, (1,1,6))  # (seq_len, batch, input_size)
        return x


    @staticmethod
    def cast_action(probs: torch.Tensor) -> int:
        expected = torch.argmax(probs, dim=2).detach().item()
        action   = int(expected + 1) % 3
        return action


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


    def forward(self, action: int, opponent: int):
        moves                 = self.cast_inputs(action, opponent)
        lstm_out, self.hidden = self.lstm(moves)
        lstm_out              = nn.functional.relu(lstm_out)
        expected_probs        = self.dense(lstm_out)
        expected_probs        = self.softmax(expected_probs)
        action                = self.cast_action(expected_probs)
        return action, expected_probs
