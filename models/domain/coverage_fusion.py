"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
MIN_SUPPORT = 10
TOL = 0.1

def mask(s, q):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    if len(s) != 20:
        raise ValueError('Exactly20 support records required')
    fields = {'cc_current_A', 'window_match'}
    if not fields <= set(s) or not fields <= set(q):
        return (np.zeros(len(q), bool), dict(support_eligible=False, reason='missing current metadata'))
    a = s.cc_current_A.to_numpy(float)
    ok = (s.window_match.to_numpy() == 1) & np.isfinite(a) & (a > 0)
    a = a[ok]
    median = float(np.median(a)) if len(a) else np.nan
    spread = float((np.quantile(a, 0.9) - np.quantile(a, 0.1)) / median) if len(a) else np.nan
    eligible = bool(len(a) >= MIN_SUPPORT and spread <= TOL)
    v = q.cc_current_A.to_numpy(float)
    valid = (q.window_match.to_numpy() == 1) & np.isfinite(v) & (v > 0)
    active = valid & (abs(v / median - 1) <= TOL) if eligible else np.zeros(len(q), bool)
    return (active, dict(support_eligible=eligible, support_valid_current_records=len(a), support_median_current_A=median, support_relative_p90_p10=spread, query_active_records=int(active.sum()), query_records=len(q), reason='eligible' if eligible else 'too few valid supports or mixed support current'))
