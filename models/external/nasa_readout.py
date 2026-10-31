"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
import numpy as np
import torch
from configs.schema import 特征列
from models.external.nasa_support import check_features
FEATURES = 特征列 + ['cycle_index']
RAW_INDICES = [i for i, c in enumerate(FEATURES) if 'kurtosis' not in c and 'skewness' not in c]

def hierarchical_head(sx, qx, y, sids, qids, prior, bias, strength, tau):
    """Adapt a prior slope in support-centered coordinates; never use query stats."""
    sx, qx, y, prior = (sx.double(), qx.double(), y.double(), prior.double())
    bias = torch.as_tensor(bias, device=sx.device, dtype=torch.float64)
    ids = sorted(set(sids))
    if set(qids) - set(ids):
        raise ValueError('Battery lacks support observations')
    centers, offsets, counts = ({}, {}, {})
    for b in ids:
        mask = torch.tensor([x == b for x in sids], device=sx.device)
        centers[b], offsets[b], counts[b] = (sx[mask].mean(0), y[mask].mean(), int(mask.sum()))
    local = sx - torch.stack([centers[b] for b in sids])
    target = y - torch.stack([offsets[b] for b in sids])
    scale = local.std(0, correction=0).clamp_min(0.1)
    z = (local / scale).clamp(-10, 10)
    residual = target - local @ prior
    delta = torch.linalg.solve(z.T @ z / len(z) + strength * torch.eye(z.shape[1], device=z.device), z.T @ residual / len(z))
    qlocal = qx - torch.stack([centers[b] for b in qids])
    qz = (qlocal / scale).clamp(-10, 10)
    shift = {b: (offsets[b] - bias - centers[b] @ prior) * counts[b] / (counts[b] + tau) for b in ids}
    pred = qx @ prior + bias + qz @ delta + torch.stack([shift[b] for b in qids])
    fit = {'prior': prior.detach().cpu().tolist(), 'bias': float(bias.detach()), 'delta': delta.detach().cpu().tolist(), 'scale': scale.detach().cpu().tolist(), 'centers': {b: v.detach().cpu().tolist() for b, v in centers.items()}, 'shifts': {b: float(v.detach()) for b, v in shift.items()}, 'counts': counts}
    return (pred, fit)

def predict_hierarchy(features, x, fit):
    check_features(features)
    ids = features.battery_id.astype(str).tolist()
    if set(ids) - set(fit['centers']):
        raise ValueError('Battery lacks support observations')
    local = x - np.asarray([fit['centers'][b] for b in ids])
    z = np.clip(local / np.asarray(fit['scale']), -10, 10)
    return x @ np.asarray(fit['prior']) + fit['bias'] + z @ np.asarray(fit['delta']) + np.asarray([fit['shifts'][b] for b in ids])
