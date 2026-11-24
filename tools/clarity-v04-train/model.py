"""Residual-only deployment graph and native-luma training adapter."""
import torch
from torch import nn
from torch.nn import functional as F

SPECS = {"Packed8": (4,8,8,4), "Packed16": (4,16,16,16,4), "Unpacked8": (1,8,8,1)}
COUNTS = {"Packed8": 1172, "Packed16": 5812, "Unpacked8": 737}
CAP = 4.0
HEADROOM = 876.0


class LumaResidual(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        channels = SPECS[kind]
        layers = []
        for i, (a,b) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.Conv2d(a,b,3,padding=1,bias=True))
            if i < len(channels)-2:
                layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        assert sum(p.numel() for p in self.parameters()) == COUNTS[kind]

    def forward(self, features):
        return self.net(features)

    def native(self, x):
        if self.kind == "Unpacked8":
            return self(x)
        return F.pixel_shuffle(self(F.pixel_unshuffle(x,2)),2)
