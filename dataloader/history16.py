"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import pandas as pd
import torch as torch
import hashlib as hashlib
from models.domain import coverage_fusion as gf
COLS = ['lowQ_slope16', 'CCQ_slope16', 'temperature_slope16', 'lowQ_current_minus_mean16', 'CCQ_current_minus_mean16', 'temperature_current_minus_mean16']
INPUT = ['battery_id', 'cycle_index', 'native_cycle', 'cc_current_A', 'window_match', 'low_window_valid', 'low38_40_Q', 'CC Q', 'cc_temperature_mean']
_CACHE = {}

def features(observed):
    if {'soh', 'capacity', 'true_soh'} & set(observed):
        raise ValueError('Query labels forbidden in history')
    result = []
    for bid, g in observed.groupby('battery_id', sort=True):
        g = g.sort_values('cycle_index')
        s = g.iloc[:20]
        if len(s) != 20:
            raise ValueError('20 observed support records required')
        _, qa = gf.mask(s, g.iloc[:0])
        out = np.full((len(g), 6), np.nan)
        if qa['support_eligible']:
            current = qa['support_median_current_A']
            v = g.cc_current_A.to_numpy(float)
            match = (g.window_match.to_numpy() == 1) & np.isfinite(v) & (v > 0) & (abs(v / current - 1) <= 0.1)
            low = g.low38_40_Q.to_numpy(float)
            cc = g['CC Q'].to_numpy(float)
            temp = g.cc_temperature_mean.to_numpy(float)
            time = g.native_cycle.to_numpy(float)
            valid_low = match & g.low_window_valid.eq(1).to_numpy(bool) & np.isfinite(low)
            valid_cc = match & np.isfinite(cc)
            valid_temp = match & np.isfinite(temp)
            ref_low = float(np.median(low[:20][valid_low[:20]])) if valid_low[:20].sum() >= 8 else np.nan
            ref_cc = float(np.median(cc[:20][valid_cc[:20]])) if valid_cc[:20].sum() >= 8 else np.nan
            series = [low / max(abs(ref_low), 1e-06) if np.isfinite(ref_low) else np.full(len(low), np.nan), cc / max(abs(ref_cc), 1e-06) if np.isfinite(ref_cc) else np.full(len(cc), np.nan), temp]
            valid = [valid_low, valid_cc, valid_temp]
            for i in range(len(g)):
                start = max(0, i - 15)
                indices = np.arange(start, i + 1)
                for k, (x, good) in enumerate(zip(series, valid)):
                    ix = indices[good[indices] & np.isfinite(x[indices]) & np.isfinite(time[indices])]
                    if len(ix) < 8:
                        continue
                    t = time[ix] - np.mean(time[ix])
                    den = float(t @ t)
                    if den > 1e-12:
                        out[i, k] = float(t @ (x[ix] - np.mean(x[ix])) / den)
                    if good[i] and np.isfinite(x[i]):
                        out[i, k + 3] = float(x[i] - np.mean(x[ix]))
        local = g[['battery_id', 'cycle_index']].copy()
        for k, col in enumerate(COLS):
            local[col] = out[:, k]
        result.append(local)
    return pd.concat(result, ignore_index=True)

def cached(observed):
    if {'soh', 'capacity', 'true_soh'} & set(observed):
        raise ValueError('Query labels forbidden in history')
    key = hashlib.sha256(pd.util.hash_pandas_object(observed[INPUT], index=True).values.tobytes()).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = features(observed)
    return _CACHE[key]

def augment(tasks, frame, norm, real=True):
    h = cached(frame.drop(columns=['soh', 'capacity', 'true_soh'], errors='ignore')).set_index(['battery_id', 'cycle_index'])
    mean = np.asarray(norm['mean'])
    sd = np.asarray(norm['sd'])
    result = []
    for original in tasks:
        t = dict(original)
        g = frame[frame.battery_id == t['battery_id']].sort_values('cycle_index')
        ids = g.cycle_index.iloc[:20].tolist() + t['query'].cycle_index.tolist()
        keys = pd.MultiIndex.from_arrays([[t['battery_id']] * len(ids), ids], names=h.index.names)
        x = h.loc[keys, COLS].to_numpy(float)
        missing = ~np.isfinite(x)
        z = np.clip((np.where(missing, mean, x) - mean) / sd, -10, 10)
        extra = np.column_stack([z, missing.astype(float)]) if real else np.zeros((len(x), 12))
        assert original['tensor'].shape[1] == 226
        t['tensor'] = torch.cat([original['tensor'], torch.from_numpy(extra.astype(np.float32)).to('cuda')], dim=1)
        result.append(t)
    return result
