# NOTE: Untested and unrun code

import torch
import torch.nn as nn

class GameOfLifeRules(nn.Module):
    """ Implement Game of Life Forward using probability rules """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.masks    = torch.FloatTensor([ list(map(int,f'{n:09b}')) for n in range(2**9) ]).reshape(1,512,3,3).to(self.device)
        self.weights  = (self.masks - 0.5) * 2.0  # cast to +1.0 || -1.0
        self.is_sum   = [ torch.sum( self.masks, dim=(2,3) ) == float(n) for n in range(10) ]
        self.is_alive = self.masks[:,:,1,1] == 1.0
        self.is_alive_future = (self.is_alive & (self.is_sum[3] | self.is_sum[4])) | (~self.is_alive & self.is_sum[3])

        self.or_conditions = [
            (i3,i4)
            for i3 in self.is_alive_future.shape[0]
            for i4 in self.is_alive_future.shape[0]
            if i3 != i4
            if torch.sum(self.is_alive_future[i3]) == 3
            if torch.sum(self.is_alive_future[i4]) == 4
            if torch.sum(self.is_alive_future[i4].bool() | self.is_alive_future[i3].bool()) == 4
        ]

        self.alive_probs = nn.Conv2d(in_channels=1, out_channels=torch.sum(self.is_alive_future).item(),
                                     kernel_size=(3,3), padding=1, padding_mode='circular', bias=False )
        self.alive_probs.weight.data = self.weights[ self.is_alive_future ].reshape(-1,1,3,3)
        self.alive_probs.weight.requires_grad_(False)
        self.to(self.device)


    def forward(self, x):
        # Probability AND rule: P(A AND B) = P(A) * P(B)
        # Probability OR  rule: P(A OR  B) = P(A) + P(B) - P(A AND B)
        input = x
        x = self.alive_probs(x)      # (batch_size, channels, height, width)
        x[ x < 0 ] = 1 - x[ x < 0 ]  # convert negative values to 1-x

        and_probs = torch.prod(x.flatten(dim=2), dim=2)  # (batch_size, channel, height * width)
        or_probs  = torch.stack([
            and_probs[:,i3,:] * and_probs[:,i4,:]
            for i3, i4 in self.or_conditions
        ], dim=1)
        or_probs = torch.sum(or_probs, dim=1)

        prob = and_probs - or_probs
        prob = prob.unsqueeze(dim=1)  # (batch_size, 1, height, width)
        return prob
