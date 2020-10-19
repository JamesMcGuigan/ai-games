# NOTE: Untested and unrun code

import torch
import torch.nn as nn

class GameOfLifeRules(nn.Module):
    """ Implement Game of Life Forward using probability rules """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        masks    = torch.IntTensor([ list(map(int,f'{n:09b}')) for n in range(2**9) ]).reshape(1,512,3,3)
        weights  = (masks.float() - 0.5) * 2.0  # cast (1,0) to (+1.0,-1.0)
        is_sum   = torch.BoolTensor([ torch.sum( masks, dim=(2,3) ) == n for n in range(10) ])
        is_alive = masks[:,:,1,1] == 1
        is_alive_future = (is_alive & (is_sum[3] | is_sum[4])) | (~is_alive & is_sum[3])


        # Create list of overlapping masks - all other combinations are mutually exclusive
        self.or_conditions = {
            (i3,i4)
            for i3 in masks.shape[0] if is_sum[3][i3]  # mutually exclusive with is_sum[4]
            for i4 in masks.shape[0] if is_sum[4][i4]  # mutually exclusive with is_sum[3]
            if torch.all( ( masks[i4] & masks[i3] ).eq( masks[i3] ) )
        }

        self.in_channels  = 1
        self.out_channels = weights[is_alive_future].shape[0]

        self.alive_probs = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                     kernel_size=(3,3), padding=1, padding_mode='circular', bias=False )
        self.alive_probs.weight.data = weights[ is_alive_future ].reshape(-1,1,3,3)
        self.alive_probs.weight.requires_grad_(False)
        self.to(self.device)


    def forward(self, x):
        # Probability AND rule: P(A AND B) = P(A) * P(B)
        # Probability OR  rule: P(A OR  B) = P(A) + P(B) - P(A AND B)
        input = x
        x = self.alive_probs(x)      # (batch_size, channels, height, width)
        x[ x < 0 ] = 1 - x[ x < 0 ]  # convert negative values to 1-x

        and_probs = torch.prod(x, dim=1)       # (batch_size, height, width)
        or_probs  = torch.stack([
            and_probs[:,i3,:] * and_probs[:,i4,:]
            for i3, i4 in self.or_conditions
        ], dim=1)                              # (batch_size, channels, height, width)
        or_probs = torch.sum(or_probs, dim=1)  # (batch_size, height, width)

        prob = and_probs - or_probs   # (batch_size, height, width)
        prob = prob.unsqueeze(dim=1)  # (batch_size, 1, height, width)
        return prob
