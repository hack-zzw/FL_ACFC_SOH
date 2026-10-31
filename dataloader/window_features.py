"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
CORE = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy', 'cycle_index']
RAW_COLUMNS = {'NCA|TJU': ['window_match', 'native_delta_cycle', 'cc_points', 'cv_points', 'cc_v_span', 'cc_current_A', 'cc_current_cv', 'cc_voltage_time_curvature', 'cc_time_fraction_0', 'cc_time_fraction_1', 'cc_time_fraction_2', 'cc_q_fraction_0', 'cc_q_fraction_1', 'cc_q_fraction_2', 'cc_integrated_Ah', 'cv_integrated_Ah', 'cc_counter_integral_difference', 'cv_counter_integral_difference', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse', 'cv_log_i_curvature', 'cc_temperature_mean', 'cv_temperature_mean', 'cc_temperature_change', 'cv_temperature_change', 'cc_max_gap_s', 'cv_max_gap_s'], 'NCM|XJTU': ['window_match', 'native_delta_cycle', 'cc_points', 'cv_points', 'cc_v_span', 'cc_current_A', 'cc_current_cv', 'cc_voltage_time_curvature', 'cc_time_fraction_0', 'cc_time_fraction_1', 'cc_time_fraction_2', 'cc_q_fraction_0', 'cc_q_fraction_1', 'cc_q_fraction_2', 'cc_integrated_Ah', 'cv_integrated_Ah', 'cc_counter_integral_difference', 'cv_counter_integral_difference', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse', 'cv_log_i_curvature', 'cc_temperature_mean', 'cv_temperature_mean', 'cc_temperature_change', 'cv_temperature_change', 'cc_max_gap_s', 'cv_max_gap_s'], 'LFP|HUST': ['window_match', 'native_delta_cycle', 'cc_points', 'cv_points', 'cc_v_span', 'cc_current_A', 'cc_current_cv', 'cc_voltage_time_curvature', 'cc_time_fraction_0', 'cc_time_fraction_1', 'cc_time_fraction_2', 'cc_q_fraction_0', 'cc_q_fraction_1', 'cc_q_fraction_2', 'cc_integrated_Ah', 'cv_integrated_Ah', 'cc_counter_integral_difference', 'cv_counter_integral_difference', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse', 'cv_log_i_curvature', 'cc_temperature_mean', 'cv_temperature_mean', 'cc_temperature_change', 'cv_temperature_change', 'cc_max_gap_s', 'cv_max_gap_s'], 'LFP|MIT': ['window_match', 'native_delta_cycle', 'protocol_C1', 'protocol_switch_SOC', 'protocol_C2', 'cc_points', 'cv_points', 'cc_start_v', 'cc_end_v', 'cc_v_span', 'cc_dt_max_s', 'cv_dt_max_s', 'cc_i_median', 'cc_i_cv', 'cv_time_high_s', 'cv_time_low_s', 'cv_time_high_fraction', 'cv_log_i_slope', 'cv_log_i_rmse', 'cc_temperature_mean', 'cv_temperature_mean', 'cc_temperature_change', 'cv_temperature_change', 'cc_bin0_time_fraction', 'cc_bin1_time_fraction', 'cc_bin2_time_fraction', 'cc_bin0_q_fraction', 'cc_bin1_q_fraction', 'cc_bin2_q_fraction']}
特征列 = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy']

def design(support, query, domain, view):
    if {'capacity', 'soh', 'true_soh'} & set(query.columns):
        raise ValueError('Query labels forbidden')
    if len(support) != 20:
        raise ValueError('Exactly20 support rows required')
    x = query[CORE].to_numpy(float)
    if view == 'original17_support_offset':
        return x
    if view != 'support_conditioned_window_shape':
        raise ValueError(view)
    s = support[特征列].to_numpy(float)
    q = query[特征列].to_numpy(float)
    ref = np.median(s, axis=0)
    relative = np.clip((q - ref) / np.maximum(abs(ref), 1e-05), -10, 10)
    sy = support.soh.to_numpy(float)
    t = support.cycle_index.to_numpy(float)
    t = t - t.mean()
    slope = float(t @ (sy - sy.mean()) / max(t @ t, 1e-12))
    context = np.r_[np.median(sy), np.ptp(sy), sy[-1] - sy[0], slope, np.median(sy[10:]) - np.median(sy[:10]), np.median(support[['CC Q', 'CV Q', 'CC charge time', 'CV charge time']], axis=0)]
    extra = query[RAW_COLUMNS[domain]].to_numpy(float)
    early_slope = np.polyfit(t[:10], sy[:10], 1)[0]
    late_slope = np.polyfit(t[10:], sy[10:], 1)[0]
    curve = np.polynomial.polynomial.polyfit(t, sy, 2)[2]
    more = np.array([sy[-1], early_slope, late_slope, late_slope - early_slope, curve])
    native_delta = query.native_cycle.to_numpy(float) - float(support.native_cycle.iloc[-1])
    extra = np.column_stack([extra, np.tile(more, (len(query), 1)), native_delta])
    return np.column_stack([x, relative, np.tile(context, (len(query), 1)), extra])
