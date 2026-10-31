"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import validation_masks as deployed
import numpy as np
from models.domain import routing as scope

def masks(cover, s, q):
    G, H, qa = scope.masks(cover, s, q)
    returned = np.zeros(len(q), bool)
    if qa['outside_temperature_coverage']:
        good, _ = deployed.mask(s, q)
        returned = good & np.isfinite(q.cc_temperature_mean.to_numpy(float)) & (q.cc_temperature_mean.to_numpy(float) <= cover['temperature_upper'])
        G |= returned
        H &= ~returned
    qa.update(returned_to_coverage_queries=int(returned.sum()), G_queries=int(G.sum()), H_queries=int(H.sum()))
    assert not (G & H).any()
    return (G, H, qa)
