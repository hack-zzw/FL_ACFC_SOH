"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np

def project(raw, anchor, lo=0.0, hi=1.1):
    return np.clip(np.minimum.accumulate(np.r_[float(anchor), np.asarray(raw, dtype=float)])[1:], lo, hi)
