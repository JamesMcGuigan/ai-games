# DOCS: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ConnectXBBNN import configuration
from neural_networks.is_gameover.BitboardNN import BitboardNN


# Linear neural network quickly reaches 95% accuracy and then plateaus
class IsGameoverSquareNN(BitboardNN):
    def __init__(self):
        super().__init__()
        self.device       = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.one_hot_size = 3
        self.input_size   = configuration.rows * configuration.columns
        self.output_size  = 1
        self.model_size   = 256

        self.fc1 = nn.Linear(self.input_size * self.one_hot_size, self.model_size)
        self.fc2 = nn.Linear(self.model_size, self.model_size)
        self.fc3 = nn.Linear(self.model_size, self.model_size)
        self.fc4 = nn.Linear(self.model_size, 1)

    def cast(self, x):
        x = super().cast(x)
        x = x.view(-1, self.input_size * self.one_hot_size)
        return x

    def forward(self, x):
        x = self.cast(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


isGameoverSquareNN = IsGameoverSquareNN()