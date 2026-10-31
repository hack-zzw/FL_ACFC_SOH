import numpy as np
import pandas as pd
import torch
DOMAINS = ('LFP|HUST', 'LFP|MIT', 'NCA|TJU', 'NCM|TJU', 'NCM|XJTU')
GROUPS = ((0, 1), (2,), (3, 4))
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
ARM = 'Mmid_material_mapping'

def normalized(x, model):
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    z = np.where(good, (x - model['mean']) / model['sd'], 0.0)
    return np.column_stack((np.clip(z, -10.0, 10.0), (~good).astype(float)))

def post(raw, anchor, variant):
    if variant == 'projected':
        raw = np.minimum.accumulate(np.r_[anchor, raw])[1:]
    elif variant != 'clip_only':
        raise ValueError(variant)
    return np.clip(raw, 0.0, 1.1)

def metrics(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    err = p - y
    den = float(np.sum((y - y.mean()) ** 2))
    return {'mae': float(np.mean(abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))), 'r2': float(1 - np.sum(err ** 2) / den) if den > 1e-15 else np.nan, 'mape': float(np.mean(abs(err) / np.maximum(abs(y), 1e-08)) * 100)}

def candidates(record):
    decompositions = []
    for group in GROUPS:
        g = record['G'][list(group)].sum(axis=0)
        h = record['h'][list(group)].sum(axis=0)
        eigenvalues, vectors = np.linalg.eigh((g + g.T) * 0.5)
        assert eigenvalues.min() > -1e-07
        decompositions.append((np.maximum(eigenvalues, 0.0), vectors, vectors.T @ h))
    result = []
    for lam in LAMBDAS:
        beta = np.empty((5, record['G'].shape[1]))
        for group, (e, v, vh) in zip(GROUPS, decompositions):
            beta[list(group)] = v @ (vh / (e + lam))
        intercept = record['my'] - np.einsum('ij,ij->i', record['mx'], beta)
        result.append({'beta': beta, 'intercept': intercept, 'mean': record['mean'], 'sd': record['sd'], 'lambda_value': lam, 'arm': ARM})
    return result

def predict_raw(model, x, task):
    i = DOMAINS.index(task['domain'])
    raw = task['support_median'] + normalized(x, model) @ model['beta'][i] + model['intercept'][i]
    delta = float(np.sum(task['support_y'] - raw[:20]) / 40.0)
    return raw[20:] + delta

def load_model(path):
    with np.load(path) as saved:
        model = {k: saved[k] for k in saved.files}
    return model
