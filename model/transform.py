from torch import nn
from model.GDN import GDN
from collections import OrderedDict


class AnalysisTransform(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.stage1 = nn.Sequential(OrderedDict([
            ("Conv9", nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(9, 9))),
            ("Downsample4", nn.AvgPool2d(kernel_size=4)),
            ("GDN", GDN(channel=256, device=device))
        ]))
        self.stage2_3 = nn.Sequential(OrderedDict([
            ("Conv5", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5))),
            ("Downsample2", nn.AvgPool2d(kernel_size=2)),
            ("GDN", GDN(channel=256, device=device))
        ]))
        self.model = nn.Sequential(OrderedDict([
            ("stage1", self.stage1),
            ("stage2", self.stage2_3),
            ("stage3", self.stage2_3),
        ]))

    def forward(self, X):
        return self.model(X)


class SynthesisTransform(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.stage1_2 = nn.Sequential(OrderedDict([
            ("IGDN", GDN(channel=256, device=device, inverse=True)),
            ("Upsample2", nn.Upsample(scale_factor=2)),
            ("Conv5", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5)))
        ]))
        self.stage3 = nn.Sequential(OrderedDict([
            ("IGDN", GDN(channel=256, device=device, inverse=True)),
            ("Upsample4", nn.Upsample(scale_factor=4)),
            ("Conv9", nn.Conv2d(in_channels=256, out_channels=3, kernel_size=(9, 9)))
        ]))
        self.model = nn.Sequential(OrderedDict([
            ("stage1", self.stage1_2),
            ("stage2", self.stage1_2),
            ("stage3", self.stage3),
            ("Resize", nn.Upsample(size=(256,256)))
        ]))

    def forward(self, y_hat):
        return self.model(y_hat)
