# DOCS: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ConnectXBBNN import configuration
from neural_networks.is_gameover.BitboardNN import BitboardNN


# self.model_size = 128 | game:  100000 | move:  2130207 | loss: 0.137 | accuracy: 0.810 / 0.953 | time: 519s
# self.model_size = 128 | game:  200000 | move:  2132342 | loss: 0.134 | accuracy: 0.834 / 0.953
# self.model_size = 128 | game: 1000000 | move: 17053998 | loss: 0.099 | accuracy: 0.890 / 0.953 | time: 4274.1s
class IsGameoverSquareNN(BitboardNN):
    def __init__(self):
        super().__init__()
        self.device       = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.one_hot_size = 3
        self.input_size   = configuration.rows * configuration.columns
        self.output_size  = 1
        self.model_size   = 128

        self.fc1    = nn.Linear(self.input_size * self.one_hot_size, self.model_size)
        self.fc2    = nn.Linear(self.model_size, self.model_size)
        self.fc3    = nn.Linear(self.model_size, self.model_size)
        self.output = nn.Linear(self.model_size, self.output_size)

    def cast(self, x):
        x = super().cast(x)
        x = x.view(-1, self.input_size * self.one_hot_size)
        return x

    def forward(self, x):
        x = self.cast(x)                   # x.shape = (1,126)
        x = F.relu(self.fc1(x))            # x.shape = (1,256)
        x = F.relu(self.fc2(x))            # x.shape = (1,256)
        x = F.relu(self.fc3(x))            # x.shape = (1,256)
        x = torch.sigmoid(self.output(x))  # x.shape = (1,1)
        x = x.view(1)                      # x.shape = (1,)  | return 1d array of outputs, to match isGameoverCNN
        return x


isGameoverSquareNN = IsGameoverSquareNN()