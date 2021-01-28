import math

import torch
import torch.nn as nn
from neural_network.NNBase import NNBase
import xxhash

# DOCS: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# DOCS: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# DOCS: https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/10-LSTM.ipynb
class RpsLSTM(NNBase):

    def __init__(self, hidden_size=128, hash_size=128, num_layers=3, dropout=0.25, window=10):
        """
        :param hidden_size: size of LSTM embedding
        :param hash_size:   size of hash for guessing opponent
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
        self.hash_size   = hash_size
        self.input_size  = self.cast_inputs(0,0).shape[-1]

        self.lstm  = nn.LSTM(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
            batch_first = True,
        )
        self.dense_1   = nn.Linear(self.input_size + hidden_size, hidden_size)
        self.dense_2   = nn.Linear(hidden_size, hidden_size)
        self.out_probs = nn.Linear(hidden_size, 3)
        self.out_hash  = nn.Linear(hidden_size, self.hash_size)

        self.activation = nn.Softsign()  # BUGFIX: PReLU was potentially causing NaNs in model weights
        self.softmax    = nn.Softmax(dim=2)
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


    def loss_probs(self, probs: torch.Tensor, opponent: int) -> torch.Tensor:
        """
        Loss based on softmax probability vs EV score of opponent move
        """
        ev = torch.zeros((3,), dtype=torch.float).to(self.device)
        ev[(opponent + 0) % 3] = 1.0   # expect rock, play paper + opponent rock     = win
        ev[(opponent + 1) % 3] = 0.5   # expect rock, play paper + opponent paper    = draw
        ev[(opponent + 2) % 3] = 0.0   # expect rock, play paper + opponent scissors = loss
        losses = probs * (1-ev)
        loss   = torch.sum( losses )
        # loss   = -torch.sum(torch.log(1-losses))  # cross entropy loss
        return loss


    def loss_hash(self, hash_id: torch.Tensor, agent_name: str) -> torch.Tensor:
        """
        Categorical Cross Entropy loss for agent prediction using Locality-sensitive hashing
        """
        hash_id    = hash_id.flatten()
        hash_pred  = torch.argmax(hash_id)
        agent_hash = xxhash.xxh32(agent_name, seed=0).intdigest() % self.hash_size
        agent_hot  = self.one_hot_encode(agent_hash, size=self.hash_size).flatten()
        loss       = -torch.sum( agent_hot * torch.log(hash_id) - (1-agent_hot) * torch.log(1-hash_id) )
        loss[ torch.isnan(loss) ] = 0.0  # BUGFIX: prevent log(0) = NaN
        return loss



    ### Casting

    def one_hot_encode(self, number: int = None, size: int = 3) -> torch.Tensor:
        """ One hot encoding of action and opponent action """
        x = torch.zeros((size,), dtype=torch.float).to(self.device)
        if number is not None:
            x[ int(number) % size ] = 1.0
        return x


    def encode_actions(self, action: int, opponent: int) -> torch.Tensor:
        return torch.stack([
            self.one_hot_encode(action),
            self.one_hot_encode(opponent),
        ])


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
        """
        Encode the step (current turn number) as one hot encoded version of the logarithm
        This is mostly for detecting random warmup periods
        { round(math.log(n)): n for n in range(1,1001)  } = { 0: 1, 1: 4, 2: 12, 3: 33, 4: 90, 5: 244, 6: 665, 7: 1000}
        """
        step     = torch.sum(self.stats[0]).reshape(1)
        log_step = torch.log(step+1).round().int().item()  # log(0) == NaN
        hot_step = self.one_hot_encode(log_step, size=round(math.log(1000)))
        return hot_step


    @staticmethod
    def cast_action(probs: torch.Tensor) -> int:
        expected = torch.argmax(probs, dim=2).detach().item()
        action   = int(expected + 1) % 3
        return action


    def cast_inputs(self, action: int, opponent: int) -> torch.Tensor:
        """
        Generate the input tensors for the LSTM
        Assumes that self.update_state(action, opponent) has been called beforehand
        action + opponent are now encoded as part of the history
        """
        if not hasattr(self, 'stats'): self.reset()

        noise   = torch.rand((3,)).to(self.device)
        step    = self.encode_step()
        stats   = self.encode_stats()
        history = self.encode_history()

        x = torch.cat([
            noise.flatten(),    # noise to simulate randomness
            step.flatten(),     # predict warmup periods
            stats.flatten(),    # for statistical bots
            history.flatten(),  # timeseries history
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
        inputs    = self.cast_inputs(action, opponent)
        x, hidden = self.lstm(inputs)
        x         = torch.cat([ x, inputs ], dim=2)
        x         = self.activation( self.dense_1(x)   )
        x         = self.activation( self.dense_2(x)   )
        probs     = self.softmax(    self.out_probs(x) )
        hash_id   = self.softmax(    self.out_hash(x)  )
        action    = self.cast_action(probs)

        # BUGFIX: occasionally LSTM would return NaNs after 850+ epochs
        #         this possibly caused by PReLU, now replaced with Softsign
        if any([ torch.any(torch.isnan(layer)) for layer in [ x, *hidden] ]):
            raise ValueError(f'{self.__class__.__name__}.forward() - LSTM returned nan')

        self.hidden = hidden
        return action, probs, hash_id
