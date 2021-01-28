import torch
import torch.nn as nn
from neural_network.NNBase import NNBase

# DOCS: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# DOCS: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# DOCS: https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/10-LSTM.ipynb
class RpsLSTM(NNBase):

    def __init__(self, hidden_size=128, num_layers=3, dropout=0.25, window=10):
        """
        :param hidden_size: size of LSTM embedding
        :param num_layers:  number of LSTM layers
        :param dropout:     dropout parameter for LSTM
        :param window:      maximum history length passed in as input data
        """
        super().__init__()
        self.device      = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.batch_size  = 1
        self.dropout     = dropout
        self.window      = window
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


    ### Lifecycle

    def reset(self):
        self.history = [ [], [] ]  # action, opponent
        self.stats   = torch.zeros((2,3),    dtype=torch.float).to(self.device)
        self.hidden  = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )


    ### Training

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
        losses = probs * (1-ev)
        loss   = -torch.sum(torch.log(1-losses))  # cross entropy loss
        return loss


    ### Casting

    def encode_actions(self, action: int = None, opponent: int = None) -> torch.Tensor:
        """ One hot encoding of action and opponent action """
        x = torch.zeros((2, 3), dtype=torch.float).to(self.device)
        if action is not None:
            x[ 0, action % 3 ] = 1.0
        return x


    def encode_history(self) -> torch.Tensor:
        """
        self.history as a one hot encoded tensor
        history is created via .insert() thus latest move will always be in position 0
        self.window can be used to restrict the maximum size of history data passed into the model
        """
        x = torch.zeros((2, self.window, 3), dtype=torch.float).to(self.device)
        for player in [0,1]:
            window = min( self.window, len(self.history[player]) )
            for step in range(window):
                action = self.history[player][step]
                if action is None: continue
                x[ player, step, action % 3 ] = 1.0
        return x


    def encode_stats(self):
        """ Normalized percentage frequency from self.stats """
        step  = torch.sum(self.stats[0])
        stats = ( self.stats / torch.sum(self.stats[0])
                  if step.item() > 0
                  else self.stats )
        return stats


    def encode_step(self):
        step = torch.sum(self.stats[0]).reshape(1)
        return step


    @staticmethod
    def cast_action(probs: torch.Tensor) -> int:
        expected = torch.argmax(probs, dim=2).detach().item()
        action   = int(expected + 1) % 3
        return action


    def cast_inputs(self, action: int, opponent: int) -> torch.Tensor:
        if not hasattr(self, 'stats'): self.reset()

        noise   = torch.rand((3,)).to(self.device)
        # step    = self.encode_step()
        stats   = self.encode_stats()
        # actions = self.encode_actions(action, opponent)
        history = self.encode_history()

        x = torch.cat([
            noise.flatten(),      # we will need noise to fight statistical bots
            # step.flatten(),
            stats.flatten(),
            # actions.flatten(),  # data also provided by history
            history.flatten(),
        ])
        x = torch.reshape(x, (1,1,-1))    # (seq_len, batch, input_size)
        return x



    ### Play

    def update_state(self, action: int, opponent: int):
        """
        self.stats records total count for each action
            which will later be normalized as percentage frequency
        self.history records move history
            [0] index always being the most recent move
        """
        if action   is not None: self.stats[0][action]   += 1.0
        if opponent is not None: self.stats[1][opponent] += 1.0
        if action   is not None: self.history[0].insert(0, action)
        if opponent is not None: self.history[1].insert(0, opponent)


    def forward(self, action: int, opponent: int):
        self.update_state(action, opponent)
        moves                 = self.cast_inputs(action, opponent)
        lstm_out, self.hidden = self.lstm(moves)
        lstm_out              = nn.functional.relu(lstm_out)
        expected_probs        = self.dense(lstm_out)
        expected_probs        = self.softmax(expected_probs)
        action                = self.cast_action(expected_probs)
        return action, expected_probs
