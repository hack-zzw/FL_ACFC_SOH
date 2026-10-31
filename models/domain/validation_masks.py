"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import coverage_fusion as gf

def mask(s, q):
    active, qa = gf.mask(s, q)
    support_good = 'low_window_valid' in s and int(s.low_window_valid.eq(1).sum()) >= 10
    if not support_good or 'low_window_valid' not in q:
        active[:] = False
    else:
        active &= q.low_window_valid.eq(1).fillna(False).to_numpy(bool)
    return (active, dict(**qa, support_low_valid=int(s.low_window_valid.eq(1).sum()) if 'low_window_valid' in s else 0, deployed_active_queries=int(active.sum()), deployed_fallback_queries=int((~active).sum())))
