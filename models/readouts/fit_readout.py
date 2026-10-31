"""Source-only ridge selection; M1 must finish before M2. No target scores."""
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import torch
from common import *

def row_weights(tasks, holdout):
    n = max((t['end'] for t in tasks))
    w = np.zeros(n)
    y = np.zeros(n)
    train = [t for t in tasks if holdout is None or t['fold'] != holdout]
    counts = {d: sum((t['domain'] == d for t in train)) for d in DOMAINS}
    for t in train:
        a, b = (t['start'] + 20, t['end'])
        w[a:b] = 1 / (5 * counts[t['domain']] * (b - a))
        y[a:b] = t['query_y'] - t['support_median']
    assert abs(w.sum() - 1) < 1e-12
    return (w, y, train)

def fit_normalization(x, w):
    d = x.shape[1]
    s0 = np.zeros(d)
    s1 = np.zeros(d)
    s2 = np.zeros(d)
    for a in range(0, len(x), 8192):
        b = min(a + 8192, len(x))
        active = w[a:b] > 0
        v = x[a:b][active].astype(float)
        weight = w[a:b][active]
        good = np.isfinite(v)
        v = np.where(good, v, 0.0)
        s0 += weight @ good
        s1 += weight @ v
        s2 += weight @ (v * v)
    mean = np.divide(s1, s0, out=np.zeros_like(s1), where=s0 > 0)
    var = np.divide(s2, s0, out=np.zeros_like(s2), where=s0 > 0) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    sd[sd < 1e-08] = 1.0
    return dict(mean=mean, sd=sd)

def stats_for_fold(x, tasks, seed, holdout):
    tag = 'all' if holdout is None else f'fold{holdout}'
    dest = OUT / 'source_statistics' / f'{seed}_{tag}.npz'
    if dest.exists():
        saved = np.load(dest)
        return {k: saved[k] for k in saved.files}
    w, y, train = row_weights(tasks, holdout)
    norm = fit_normalization(x, w)
    dim = 2 * x.shape[1]
    Sxx = np.zeros((5, dim, dim))
    Sxy = np.zeros((5, dim))
    Sx = np.zeros((5, dim))
    Sy = np.zeros(5)
    mass = np.zeros(5)
    audit = []
    for client in sorted(set((t['client_id'] for t in train))):
        group = [t for t in train if t['client_id'] == client]
        d = DOMAINS.index(group[0]['domain'])
        assert all((t['domain'] == DOMAINS[d] for t in group))
        count = 0
        client_mass = 0.0
        for t in group:
            for a in range(t['start'] + 20, t['end'], 4096):
                b = min(a + 4096, t['end'])
                z = normalized(x[a:b], norm)
                tx = torch.from_numpy(z).to('cuda')
                tw = torch.from_numpy(w[a:b]).to('cuda')
                ty = torch.from_numpy(y[a:b]).to('cuda')
                wx = tx * torch.sqrt(tw[:, None])
                Sxx[d] += (wx.T @ wx).cpu().numpy()
                Sxy[d] += (tx.T @ (tw * ty)).cpu().numpy()
                Sx[d] += (tx.T @ tw).cpu().numpy()
                Sy[d] += float(tw @ ty)
                mass[d] += float(tw.sum())
                count += b - a
                client_mass += float(tw.sum())
        audit.append(dict(seed=seed, normalization_fold=tag, client_id=client, domain=DOMAINS[d], training_batteries=len(group), query_records=count, total_weight=client_mass, individual_query_observations_exported=False, aggregation='single-process sum of client weighted sufficient statistics'))
    assert np.allclose(mass, 0.2, atol=1e-12)
    mx = Sx / mass[:, None]
    my = Sy / mass
    G = Sxx - mass[:, None, None] * (mx[:, :, None] * mx[:, None, :])
    h = Sxy - mass[:, None] * mx * my[:, None]
    record = dict(G=G, h=h, mx=mx, my=my, mean=norm['mean'], sd=norm['sd'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **record)
    pd.DataFrame(audit).to_csv(dest.with_suffix('.clients.csv'), index=False)
    return record

def candidate_models(record, arm):
    if arm == 'M1_shared_mapping':
        Gs = [sum(record['G'])]
        hs = [sum(record['h'])]
    else:
        Gs = list(record['G'])
        hs = list(record['h'])
    eigen = []
    for G, h in zip(Gs, hs):
        e, v = np.linalg.eigh((G + G.T) * 0.5)
        assert e.min() > -1e-07, float(e.min())
        eigen.append((np.maximum(e, 0), v, v.T @ h))
    result = []
    for lam in LAMBDAS:
        beta = [v @ (vh / (e + lam)) for e, v, vh in eigen]
        if arm == 'M1_shared_mapping':
            beta = beta * 5
        beta = np.stack(beta)
        intercept = record['my'] - np.einsum('ij,ij->i', record['mx'], beta)
        result.append(dict(beta=beta, intercept=intercept, normalization=dict(mean=record['mean'], sd=record['sd']), lambda_value=lam, arm=arm))
    return result

def validate_candidates(models, x, task, seed, arm):
    z = normalized(x, models[0]['normalization'])
    d = DOMAINS.index(task['domain'])
    beta = np.stack([m['beta'][d] for m in models], axis=1)
    raw = task['support_median'] + z @ beta + np.array([m['intercept'][d] for m in models])[None, :]
    delta = np.sum(task['support_y'][:, None] - raw[:20], axis=0) / 40.0
    raw = raw[20:] + delta
    return [dict(seed=seed, arm=arm, fold=task['fold'], battery_id=task['battery_id'], domain=task['domain'], lambda_value=m['lambda_value'], **metrics(task['query_y'], post(raw[:, j], task['support_last'], 'projected'))) for j, m in enumerate(models)]
