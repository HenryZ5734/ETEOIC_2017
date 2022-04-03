import torch
from model.transform import *
from model.prob_cumulator import Cumulator


class Net(nn.Module):
    def __init__(self, device, is_train):
        super(Net, self).__init__()
        self.device = device
        self.is_train = is_train
        self.net = nn.Sequential(OrderedDict([
            ("Analysis", AnalysisTransform(device=device)),
            ("Synthesis", SynthesisTransform(device=device))
        ]))

    def calculate_distortion(self, X, X_hat):
        distortion = torch.mean(torch.square(X - X_hat))
        return distortion

    def calculate_rate(self, X, Y):
        cumulative = Cumulator(Y.shape[1]).to(self.device)
        # 两个概率累积的差值即为对应点的概率
        p_y = cumulative(Y + 0.5) - cumulative(Y - 0.5)
        sum_of_bits = torch.sum(-torch.log2(p_y))
        # 这里不确定要不要除以输入图片的通道数
        return sum_of_bits / (X.shape[0] * X.shape[2] * X.shape[3])

    def forward(self, X):

        # analysis transform
        y = self.net[0](X)

        # 训练时量化会导致无法求导，故用noise模拟量化效果
        if self.is_train:
            noise = nn.init.uniform_(torch.empty(y.shape), -0.5, 0.5)
            noise = noise.to(self.device)
            q = y + noise
        else:
            q = torch.round(y)

        # 计算bpp
        rate = self.calculate_rate(X, q)

        # synthesis transform
        X_hat = self.net[1](q)

        # 计算失真率
        distortion = self.calculate_distortion(X, X_hat)

        return X_hat, rate, distortion
