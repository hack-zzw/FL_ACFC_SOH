"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import coverage_fusion as gf
import numpy as np

def masks(record, s, q):
    current, qa = gf.mask(s, q)
    G = current.copy()
    H = np.zeros(len(q), bool)
    near6 = qa['support_eligible'] and 5.4 <= qa['support_median_current_A'] <= 6.6
    info = dict(**qa, near6A=bool(near6), outside_temperature_coverage=False, support_temperature=None, valid_support_temperatures=0, valid_support_low=0, training_temperature_upper=record['temperature_upper'], reason_scope='ordinary current group')
    if near6:
        if 'cc_temperature_mean' not in s:
            G[:] = False
            info['reason_scope'] = 'missing support temperature'
        else:
            good = (s.window_match.to_numpy() == 1) & np.isfinite(s.cc_temperature_mean.to_numpy(float))
            temp = float(np.median(s.cc_temperature_mean.to_numpy(float)[good])) if good.any() else None
            count = int(good.sum())
            info.update(support_temperature=temp, valid_support_temperatures=count)
            upper = record['temperature_upper']
            if count < 10 or record['matching_current_batteries'] < 4 or upper is None or (not np.isfinite(upper)):
                G[:] = False
                info['reason_scope'] = 'insufficient temperature coverage information'
            elif temp > upper:
                G[:] = False
                info['outside_temperature_coverage'] = True
                info['reason_scope'] = 'outside training temperature coverage'
                low = int(s.low_window_valid.eq(1).sum()) if 'low_window_valid' in s else 0
                info['valid_support_low'] = low
                if low >= 10 and {'low_window_valid', 'cc_temperature_mean'} <= set(q):
                    H = current & q.low_window_valid.eq(1).fillna(False).to_numpy(bool) & np.isfinite(q.cc_temperature_mean.to_numpy(float))
            else:
                info['reason_scope'] = 'inside training temperature coverage'
    info.update(G_queries=int(G.sum()), H_queries=int(H.sum()), fallback_queries=int((~(G | H)).sum()))
    assert not (G & H).any()
    return (G, H, info)
