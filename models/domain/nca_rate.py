"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import torch.nn as nn

class Residual(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4), nn.SiLU(), nn.Linear(4, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)
