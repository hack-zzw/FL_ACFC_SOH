"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import temperature_masks as hard
import numpy as np

def weight(coverage, sd, s, q):
    G, H, qa = hard.masks(coverage, s, q)
    a = np.zeros(len(q))
    if qa['outside_temperature_coverage']:
        a[G] = np.clip((coverage['temperature_upper'] - q.cc_temperature_mean.to_numpy(float)[G]) / sd, 0, 1)
    return a
