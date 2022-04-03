from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, rate, distortion, d_lambda):
        return rate + d_lambda * distortion
