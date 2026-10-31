"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
QUALITY = ['window_match', 'protocol_C1', 'protocol_switch_SOC', 'protocol_C2', 'cc_points', 'cv_points', 'cc_v_span', 'cc_dt_max_s', 'cv_dt_max_s', 'cc_i_median', 'cc_i_cv', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse']

def design(support, query, branch):
    if {'soh', 'capacity', 'true_soh'} & set(query):
        raise ValueError('Query labels forbidden')
    if len(support) != 20:
        raise ValueError('Exactly20support labels')
    np.testing.assert_array_equal(query.cycle_index, branch.cycle_index)
    quality = query[QUALITY].to_numpy(float)
    core = ['CC Q', 'CV Q', 'CC charge time', 'CV charge time']
    reference = np.median(support[core], axis=0)
    relative = np.clip((query[core].to_numpy() - reference) / np.maximum(abs(reference), 1e-06), -10, 10)
    ar = branch.acfc_raw.to_numpy(float)
    rr = branch.reference_raw.to_numpy(float)
    mixed = 0.5 * (ar + rr)
    current = branch.incumbent.to_numpy(float)
    gap = mixed - current
    difference = ar - rr
    points = np.maximum(query.cc_points.to_numpy(float), 1.0)
    window = np.maximum(query.cc_v_span.to_numpy(float), 0.0001)
    context = np.tile([np.median(support.soh), np.ptp(support.soh), support.soh.iloc[-1] - support.soh.iloc[0]], (len(query), 1))
    interactions = np.column_stack([ar, rr, current, gap, difference, abs(difference), gap / (points + 5), difference / (points + 5), gap / window, difference / window])
    return np.column_stack([quality, relative, context, interactions])

def normalize(x, record=None):
    missing = ~np.isfinite(x)
    if record is None:
        n = (~missing).sum(axis=0)
        mean = np.where(missing, 0, x).sum(axis=0) / np.maximum(n, 1)
        sd = np.maximum(np.sqrt(np.where(missing, 0, (x - mean) ** 2).sum(axis=0) / np.maximum(n, 1)), 1e-05)
        record = dict(mean=mean.tolist(), sd=sd.tolist())
    mean = np.array(record['mean'])
    sd = np.array(record['sd'])
    z = np.clip((np.where(missing, mean, x) - mean) / sd, -10, 10)
    return (np.column_stack([np.ones(len(x)), z, missing.astype(float)]), record)

def correct(record, support, query, branch):
    x = design(support, query, branch)
    z, _ = normalize(x, record['normalization'])
    correction = 0.03 * np.tanh(z @ np.array(record['coefficients']) / 0.03)
    return np.clip(branch.incumbent.to_numpy(float) + correction, 0, 1.1)
