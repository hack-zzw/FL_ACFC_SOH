"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import pandas as pd
HISTORY_COLS = ['CC Q', 'CV Q', 'CC charge time', 'CV charge time', 'cc_voltage_time_curvature', 'cv_log_i_slope', 'cc_temperature_mean', 'cv_temperature_mean']

def history_features(features):
    """Current/past observed windows only; no SOH, capacity or future rows."""
    if {'soh', 'capacity', 'true_soh'} & set(features):
        raise ValueError('Query labels forbidden')
    rows = []
    for bid, g in features.groupby('battery_id', sort=True):
        g = g.sort_values('cycle_index').copy()
        t = g.native_cycle.astype(float)
        result = g[['battery_id', 'cycle_index']].copy()
        for lag in (1, 3):
            dt = t - t.shift(lag)
            dt = dt.where(dt > 0, np.nan)
            for col in HISTORY_COLS:
                result[f'history_{col}_lag{lag}'] = (g[col] - g[col].shift(lag)) / dt
        rows.append(result)
    return pd.concat(rows, ignore_index=True)
