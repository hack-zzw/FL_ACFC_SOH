"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
SHAPE_COLUMNS = ['window_match', 'native_delta_cycle', 'cc_points', 'cv_points', 'cc_v_span', 'cc_current_A', 'cc_current_cv', 'cc_voltage_time_curvature', 'cc_time_fraction_0', 'cc_time_fraction_1', 'cc_time_fraction_2', 'cc_q_fraction_0', 'cc_q_fraction_1', 'cc_q_fraction_2', 'cc_integrated_Ah', 'cv_integrated_Ah', 'cc_counter_integral_difference', 'cv_counter_integral_difference', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse', 'cv_log_i_curvature', 'cc_temperature_mean', 'cv_temperature_mean', 'cc_temperature_change', 'cv_temperature_change', 'cc_max_gap_s', 'cv_max_gap_s']

def design(support, query, mixed_raw, current_final, original_raw, original_final):
    if {'soh', 'capacity', 'true_soh'} & set(query):
        raise ValueError('Query labels forbidden')
    if len(support) != 20:
        raise ValueError('Exactly20 support rows required')
    core = ['CC Q', 'CV Q', 'CC charge time', 'CV charge time']
    q = query[core].to_numpy(float)
    ref = np.median(support[core].to_numpy(float), axis=0)
    relative = np.clip((q - ref) / np.maximum(abs(ref), 1e-06), -10, 10)
    sy = support.soh.to_numpy(float)
    context = np.tile([np.median(sy), np.ptp(sy), sy[-1] - sy[0], np.median(sy[10:]) - np.median(sy[:10])], (len(query), 1))
    raw = np.asarray(mixed_raw, float)
    current = np.asarray(current_final, float)
    ar = np.asarray(original_raw, float)
    af = np.asarray(original_final, float)
    nr = 2 * raw - ar
    nf = 2 * current - af
    native = query.native_cycle.to_numpy(float) - float(support.native_cycle.iloc[-1])
    partial_sum = q[:, 0] + q[:, 1]
    support_sum = float(np.median(support['CC Q'] + support['CV Q']))
    partial_change = np.clip((partial_sum - support_sum) / max(abs(support_sum), 1e-06), -10, 10)
    decline = np.maximum(-partial_change, 0)
    rise = np.maximum(partial_change, 0)
    disagreement = ar - nr
    gap = raw - current
    interactions = np.column_stack([current, current - np.median(sy), ar, nr, af, nf, disagreement, abs(disagreement), gap, native, partial_change, decline, decline ** 2, rise, decline * disagreement, decline * gap, relative[:, 0] * context[:, 2], relative[:, 1] * context[:, 2]])
    return np.column_stack([query[SHAPE_COLUMNS].to_numpy(float), relative, context, interactions])

def normalize(x, record=None):
    missing = ~np.isfinite(x)
    if record is None:
        count = (~missing).sum(axis=0)
        mean = np.where(missing, 0, x).sum(axis=0) / np.maximum(count, 1)
        sd = np.maximum(np.sqrt(np.where(missing, 0, (x - mean) ** 2).sum(axis=0) / np.maximum(count, 1)), 1e-05)
        record = dict(mean=mean.tolist(), sd=sd.tolist())
    mean = np.array(record['mean'])
    sd = np.array(record['sd'])
    z = np.clip((np.where(missing, mean, x) - mean) / sd, -10, 10)
    return (np.column_stack([np.ones(len(x)), z, missing.astype(float)]), record)

def correct_from_x(record, x, base):
    z, _ = normalize(x, record['normalization'])
    correction = record['cap'] * np.tanh(z @ np.array(record['coefficients']) / record['cap'])
    return np.clip(np.asarray(base, float) + correction, 0, 1.1)

def apply_correction(record, support, query, mixed_raw, current_final, original_raw, original_final):
    x = design(support, query, mixed_raw, current_final, original_raw, original_final)
    return correct_from_x(record, x, current_final)
