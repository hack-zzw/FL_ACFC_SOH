"""One scoring definition for all five domains and all three components."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
SEEDS = (20260903, 20260904, 20260905)
DOMAINS = ('LFP|HUST', 'LFP|MIT', 'NCA|TJU', 'NCM|TJU', 'NCM|XJTU')
KEY = ['battery_id', 'cycle_index']

def read(path):
    return pd.read_csv(path, float_precision='round_trip')

def metric(y, p):
    y = np.asarray(y, float)
    e = np.asarray(p, float) - y
    den = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e * e))), r2=float(1 - np.sum(e * e) / den) if den > 1e-15 else float('nan'), mape=float(np.mean(abs(e) / np.maximum(abs(y), 1e-08)) * 100))

def average_components(parts):
    assert len(parts) == 3
    parts = [f.sort_values(KEY).reset_index(drop=True) for f in parts]
    identity = KEY + ['true_soh'] + (['domain'] if 'domain' in parts[0] else [])
    for f in parts[1:]:
        pd.testing.assert_frame_equal(parts[0][identity], f[identity], check_dtype=False)
    result = parts[0].copy()
    for col in ['pred_soh', 'raw_pred_soh']:
        if all((col in f for f in parts)):
            result[col] = np.mean([f[col].to_numpy() for f in parts], axis=0)
    if 'seed' in result:
        result['seed'] = 0
    return result

def summarize(frame):
    assert not frame.duplicated(KEY).any()
    rows = []
    for bid, g in frame.groupby('battery_id', sort=True):
        rows.append(dict(battery_id=bid, domain=str(g.domain.iloc[0]), queries=len(g), **metric(g.true_soh, g.pred_soh), bias=float((g.pred_soh - g.true_soh).mean()), predicted_increases=int(np.sum(np.diff(g.sort_values('cycle_index').pred_soh) > 1e-12))))
    battery = pd.DataFrame(rows)
    rows = []
    for d, g in battery.groupby('domain', sort=True):
        rows.append(dict(domain=d, batteries=len(g), queries=int(g.queries.sum()), **{k: float(g[k].mean()) for k in ['mae', 'rmse', 'r2', 'mape']}))
    domain = pd.DataFrame(rows)
    pooled = metric(frame.true_soh, frame.pred_soh)
    pooled.update(batteries=len(battery), queries=len(frame), battery_macro_r2=float(battery.r2.mean()), battery_macro_mae=float(battery.mae.mean()))
    return (battery, domain, pooled)

def primary_intervals(battery):
    rng = np.random.default_rng(20261025)
    rows = []
    for d in DOMAINS:
        b = battery[battery.domain.eq(d)]
        ix = rng.integers(0, len(b), size=(10000, len(b)))
        for col in ['mae', 'rmse', 'r2']:
            samples = np.nanmean(b[col].to_numpy()[ix], axis=1)
            rows.append(dict(domain=d, metric=col, estimate=float(b[col].mean()), low95=float(np.quantile(samples, 0.025)), high95=float(np.quantile(samples, 0.975)), resampling_unit='whole battery', conditional_on='fixed fitted ensemble and development cohort', repetitions=10000))
    return pd.DataFrame(rows)

def assert_numeric(actual, expected, keys, columns, tolerance=1e-10):
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    maximum = 0.0
    for col in columns:
        diff = float(np.nanmax(abs(a[col].to_numpy(float) - b[col].to_numpy(float))))
        assert diff < tolerance, (col, diff, tolerance)
        maximum = max(maximum, diff)
    return maximum

def metric_roundoff_bounds(actual, reference):
    """Propagate measured prediction differences through the score definitions."""
    a = actual.sort_values(KEY).reset_index(drop=True)
    b = reference.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[KEY + ['true_soh']], b[KEY + ['true_soh']], check_dtype=False)
    y = b.true_soh.to_numpy(float)
    error = b.pred_soh.to_numpy(float) - y
    delta = abs(a.pred_soh.to_numpy(float) - b.pred_soh.to_numpy(float))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(mae=float(delta.mean()), rmse=float(np.sqrt(np.mean(delta ** 2))), r2=float(np.sum(2 * abs(error) * delta + delta ** 2) / sst), mape=float(100 * np.mean(delta / np.maximum(abs(y), 1e-08))))
