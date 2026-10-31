"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np

def project(raw, anchor):
    return np.clip(np.minimum.accumulate(np.r_[anchor, np.asarray(raw, float)])[1:], 0, 1.1)
