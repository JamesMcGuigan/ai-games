# Modified from Source: https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# Switched from using F.cross_entropy() to F.binary_cross_entropy()
# NOTE: FocalLoss() is almost exactly the same as 1-accuracy()
import torch

import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param  = balance_param
        self.bce            = nn.BCELoss()

    def forward(self, output, target):
        logpt      = - self.bce(output, target)
        pt         = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


# def test_focal_loss():
#     loss = FocalLoss()
#
#     input = Variable(torch.randn(3, 5), requires_grad=True)
#     target = Variable(torch.LongTensor(3).random_(5))
#
#     print(input)
#     print(target)
#
#     output = loss(input, target)
#     print(output)
#     output.backward()
