"""Fixed observable design and numerics for M1/M2; no query-label input."""
from __future__ import annotations
import hashlib
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.paths import ROOT as PACKAGE
import os
P = PACKAGE
OUT = Path(os.environ.get('SOH_RUN_OUTPUT', str(PACKAGE / 'results/generated'))) / 'readouts'
OUT.mkdir(parents=True, exist_ok=True)
PREP = Path(os.environ.get('SOH_FEATURE_CACHE', str(PACKAGE / 'data/prepared')))
AUDIT = PACKAGE / 'configs/readouts'
REF = PACKAGE / 'results/reference'
ABL = PACKAGE / 'results/reference/metrics/ablations'
R = P
from configs.schema import 特征列, 实验配置
from models.encoder import 按配置创建模型
from models.predict import verify, feature_tables
from utils.state import restore_scaler
from dataloader.history16 import features as history_features, COLS as HISTORY
from dataloader.qv_features import GRID, FEATURES as QV
SEEDS = (20260903, 20260904, 20260905)
DOMAINS = ['LFP|HUST', 'LFP|MIT', 'NCA|TJU', 'NCM|TJU', 'NCM|XJTU']
LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
CORE = list(特征列) + ['cycle_index']
WHITE = json.loads((AUDIT / 'matched_comparator_input_whitelist.json').read_text(encoding='utf-8'))
WINDOW = sorted(set((c for v in WHITE['observable_window_columns'].values() for c in v)))
EXTRA = WINDOW + ['experiment_temperature_C', 'low38_40_Q', 'low_window_valid']
LABELS = {'soh', 'capacity', 'true_soh'}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def finite_median(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan

def numeric_schema():
    return ['core:' + c for c in CORE] + ['relative:' + c for c in 特征列] + ['support:' + c for c in WHITE['support_context_14']] + [f'interaction:{c}:{k}' for k in ['median_soh_minus_.95', 'range_soh', 'last_minus_first_soh'] for c in 特征列] + ['window:' + c for c in EXTRA] + [f'support_window:{c}:{k}' for c in EXTRA for k in ['median', 'finite_count']] + ['native_cycle_since_support_end', 'support_native_cycle_origin'] + ['NCA_qv_increment_3.6_3.8', 'NCA_qv_increment_3.8_4.0', 'NCA_qv_increment_4.0_4.18', 'NCA_qv_amplitude', 'NCA_qv_valid'] + ['XJTU_lowQ_relative'] + ['XJTU_history:' + c for c in HISTORY]

def design(observed, support, domain):
    """May use all20 supports; each later row uses only its current/past observations."""
    if LABELS & set(observed):
        raise ValueError('Query/observed labels forbidden')
    if len(support) != 20:
        raise ValueError('Exactly20 support labels required')
    if not observed.cycle_index.is_monotonic_increasing:
        raise ValueError('Sort observed records first')
    n = len(observed)
    x = observed[CORE].to_numpy(float)
    ref = np.median(support[特征列].to_numpy(float), axis=0)
    rel = np.clip((x[:, :16] - ref) / np.maximum(abs(ref), 1e-05), -10, 10)
    y = support.soh.to_numpy(float)
    t = support.cycle_index.to_numpy(float)
    t = t - t.mean()
    slope = float(t @ (y - y.mean()) / max(t @ t, 1e-12))
    early = float(np.polyfit(t[:10], y[:10], 1)[0])
    late = float(np.polyfit(t[10:], y[10:], 1)[0])
    context = np.r_[np.median(y), np.ptp(y), y[-1] - y[0], slope, np.median(y[10:]) - np.median(y[:10]), np.median(support[['CC Q', 'CV Q', 'CC charge time', 'CV charge time']], axis=0), y[-1], early, late, late - early, np.polynomial.polynomial.polyfit(t, y, 2)[2]]
    interactions = np.column_stack([rel * c for c in [np.median(y) - 0.95, np.ptp(y), y[-1] - y[0]]])
    active = set(WHITE['observable_window_columns'][domain])
    if domain == 'NCA|TJU':
        active.add('experiment_temperature_C')
    if domain == 'NCM|XJTU':
        active.update(['low38_40_Q', 'low_window_valid'])
    current = []
    support_extra = []
    for c in EXTRA:
        if c in active and c in observed:
            a = observed[c].to_numpy(float)
            s = support[c].to_numpy(float)
            current.append(a)
            support_extra.extend([finite_median(s), float(np.isfinite(s).sum())])
        else:
            current.append(np.full(n, np.nan))
            support_extra.extend([np.nan, np.nan])
    native = observed.native_cycle.to_numpy(float) if 'native_cycle' in observed else observed.cycle_index.to_numpy(float)
    origin = float(support.native_cycle.iloc[-1]) if 'native_cycle' in support else float(support.cycle_index.iloc[-1])
    qv = np.full((n, 5), np.nan)
    if domain == 'NCA|TJU':
        sv = support[list(QV)].to_numpy(float)
        reference = np.array([finite_median(sv[:, j]) for j in range(len(QV))])
        relative = (observed[list(QV)].to_numpy(float) - reference) / max(float(reference[-1]), 1e-06)
        good = np.isfinite(relative).all(axis=1) & observed.qv_valid.fillna(False).to_numpy(bool)
        if np.isfinite(reference).all():
            mid = np.array([[np.interp(3.8, GRID, a), np.interp(4.0, GRID, a), a[-1]] for a in relative])
            qv[:, :4] = np.column_stack([mid[:, 0], mid[:, 1] - mid[:, 0], mid[:, 2] - mid[:, 1], relative[:, -1]])
            qv[~good, :4] = np.nan
        qv[:, 4] = good.astype(float)
    low = np.full(n, np.nan)
    history = np.full((n, 6), np.nan)
    if domain == 'NCM|XJTU':
        r = finite_median(support.low38_40_Q.to_numpy(float))
        if np.isfinite(r):
            good = observed.low_window_valid.fillna(0).eq(1).to_numpy(bool)
            low = (observed.low38_40_Q.to_numpy(float) - r) / max(abs(r), 1e-06)
            low[~good] = np.nan
        history = history_features(observed)[HISTORY].to_numpy(float)
    a = np.column_stack([x, rel, np.tile(context, (n, 1)), interactions, np.column_stack(current), np.tile(support_extra, (n, 1)), native - origin, np.full(n, origin), qv, low, history])
    a[~np.isfinite(a)] = np.nan
    assert a.shape == (n, len(numeric_schema())), a.shape
    return a.astype(np.float32)

def merge_observables(core, source):
    inventory = pd.read_csv(AUDIT / 'source_feature_asset_inventory.csv').set_index('role')
    inventory['path'] = [str(PACKAGE / v) for v in inventory.path]
    roles = {'LFP|HUST': 'HUST_window', 'LFP|MIT': 'MIT_window_quality', 'NCA|TJU': 'NCA_observation_linkage', 'NCM|XJTU': 'XJTU_window_canonical'}
    side = {d: pd.read_pickle(inventory.loc[role, 'path']) for d, role in roles.items()} if source else feature_tables(R)
    if source:
        qv = pd.concat([pd.read_pickle(inventory.loc[role, 'path']) for role in ['NCA_QV', 'NCA_halfC_QV']], ignore_index=True)
        side['NCA|TJU'] = side['NCA|TJU'].merge(qv[['battery_id', 'cycle_index', 'qv_valid'] + list(QV)], on=['battery_id', 'cycle_index'], how='left', validate='one_to_one')
        low = pd.read_pickle(inventory.loc['XJTU_low_voltage', 'path'])
        side['NCM|XJTU'] = side['NCM|XJTU'].merge(low[['battery_id', 'cycle_index', 'low38_40_Q', 'low_window_valid']], on=['battery_id', 'cycle_index'], how='left', validate='one_to_one')
    temperatures = json.loads((R / 'configs/nca_protocol_metadata.json').read_text(encoding='utf-8'))
    pieces = []
    for d, g in core.groupby('domain', sort=True):
        if d in side:
            fields = ['battery_id', 'cycle_index', 'native_cycle'] + WHITE['observable_window_columns'][d]
            if d == 'NCA|TJU':
                fields += ['qv_valid'] + list(QV)
            if d == 'NCM|XJTU':
                fields += ['low38_40_Q', 'low_window_valid']
            fields = list(dict.fromkeys(fields))
            g = g.merge(side[d][fields], on=['battery_id', 'cycle_index'], how='left', validate='one_to_one')
        if d == 'NCA|TJU':
            g['experiment_temperature_C'] = g.battery_id.map(temperatures)
        pieces.append(g)
    result = pd.concat(pieces, ignore_index=True).sort_values(['battery_id', 'cycle_index']).reset_index(drop=True)
    result['native_cycle'] = result.native_cycle.fillna(result.cycle_index)
    return result

def embedding(frame, seed, output_name):
    file = PREP / f'{output_name}_embedding_{seed}.npy'
    if file.exists():
        return np.load(file, mmap_mode='r')
    bundle = torch.load(R / 'checkpoints/main' / f'seed_{seed}.pt', map_location='cpu', weights_only=True)
    cfg = 实验配置(**bundle['config'])
    model = 按配置创建模型(cfg).to('cuda').eval()
    model.load_state_dict(bundle['global_state'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    scaler = restore_scaler(bundle['scaler'])
    a = scaler.变换(frame)
    values = []
    with torch.no_grad():
        for start in range(0, len(a), 4096):
            values.append(model.solution.get_embedding(torch.from_numpy(a[start:start + 4096]).to('cuda')).cpu().numpy())
    z = np.concatenate(values).astype(np.float32)
    assert z.shape == (len(frame), 36) and np.isfinite(z).all()
    np.save(file, z)
    return z

def normalized(x, norm):
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    z = np.where(good, (x - norm['mean']) / norm['sd'], 0.0)
    return np.column_stack([np.clip(z, -10, 10), (~good).astype(float)])

def post(raw, anchor, variant):
    if variant == 'projected':
        raw = np.minimum.accumulate(np.r_[anchor, np.asarray(raw)])[1:]
    elif variant != 'clip_only':
        raise ValueError(variant)
    return np.clip(raw, 0, 1.1)

def predict_task(model, x, task, variant='projected'):
    d = task['domain']
    z = normalized(x, model['normalization'])
    b = model['beta'][DOMAINS.index(d)]
    raw = task['support_median'] + z @ b + model['intercept'][DOMAINS.index(d)]
    delta = float(np.sum(task['support_y'] - raw[:20]) / 40.0)
    raw = raw[20:] + delta
    return (raw, post(raw, task['support_last'], variant))

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = p - y
    den = float(np.sum((y - y.mean()) ** 2))
    return dict(mae=float(np.mean(abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))), r2=float(1 - np.sum(err ** 2) / den) if den > 1e-15 else np.nan, mape=float(np.mean(abs(err) / np.maximum(abs(y), 1e-08)) * 100))

def model_from_stats(record, lam, arm):
    beta = []
    intercepts = []
    if arm == 'M1_shared_mapping':
        G = sum(record['G'])
        h = sum(record['h'])
        eig, vec = np.linalg.eigh((G + G.T) * 0.5)
        assert eig.min() > -1e-07, float(eig.min())
        b = vec @ (vec.T @ h / (np.maximum(eig, 0) + lam))
        beta = [b] * 5
    elif arm == 'M2_domain_mapping':
        for G, h in zip(record['G'], record['h']):
            eig, vec = np.linalg.eigh((G + G.T) * 0.5)
            assert eig.min() > -1e-07, float(eig.min())
            beta.append(vec @ (vec.T @ h / (np.maximum(eig, 0) + lam)))
    else:
        raise ValueError(arm)
    for i, b in enumerate(beta):
        intercepts.append(record['my'][i] - record['mx'][i] @ b)
    return dict(beta=np.stack(beta), intercept=np.array(intercepts), normalization=dict(mean=record['mean'], sd=record['sd']), lambda_value=lam, arm=arm)

def save_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, beta=model['beta'], intercept=model['intercept'], mean=model['normalization']['mean'], sd=model['normalization']['sd'], lambda_value=np.array(model['lambda_value']), arm=np.array(model['arm']))

def load_model(path):
    a = np.load(path)
    return dict(beta=a['beta'], intercept=a['intercept'], normalization=dict(mean=a['mean'], sd=a['sd']), lambda_value=float(a['lambda_value']), arm=str(a['arm']))
