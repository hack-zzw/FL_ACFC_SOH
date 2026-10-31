"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import torch.nn as nn
import torch as torch
import numpy as np
import pandas as pd
from dataloader import nca_window as teachers
from dataloader import qv_features as qv
CAP = 0.03

class Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 12), nn.SiLU(), nn.Linear(12, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        y = self.net(x).squeeze(-1)
        return CAP * torch.tanh((y - torch.quantile(y[:20], 0.5)) / CAP)

def shape(support, query):
    if {'soh', 'capacity', 'true_soh'} & set(query):
        raise ValueError('Query labels forbidden')
    qs = support[qv.FEATURES].to_numpy(float)
    x = query[qv.FEATURES].to_numpy(float)
    ref = np.nanmedian(qs, axis=0)
    total = max(float(ref[-1]), 1e-06)
    rel = (x - ref) / total
    amp = rel[:, -1]
    profile = rel - amp[:, None] * (qv.GRID - qv.GRID[0]) / (qv.GRID[-1] - qv.GRID[0])
    valid = np.isfinite(rel).all(axis=1) & query.qv_valid.to_numpy(bool)
    return (profile, amp, valid, rel)

def design(support, query, raw, pca, view):
    profile, amp, valid, relative = shape(support, query)
    if view == 'shape':
        three = (profile - np.array(pca['mean'])) @ np.array(pca['components']).T
    else:
        middle = np.array([[np.interp(3.8, qv.GRID, r), np.interp(4.0, qv.GRID, r), r[-1]] for r in relative])
        three = np.column_stack([middle[:, 0], middle[:, 1] - middle[:, 0], middle[:, 2] - middle[:, 1]])
    y = support.soh.to_numpy(float)
    t = support.cycle_index.to_numpy(float)
    first = float(np.polyfit(t[:10], y[:10], 1)[0])
    last = float(np.polyfit(t[10:], y[10:], 1)[0])
    context = [float(np.median(y)) - 0.9, float(np.ptp(y)), float(y[-1] - np.median(y)), last * 10, (last - first) * 10]
    features = np.column_stack([three, amp, np.tile(context, (len(query), 1)), np.asarray(raw) - float(np.median(y))])
    return (features, valid)

def inference(record, support, query, raw_support, raw_query):
    if {'soh', 'capacity', 'true_soh'} & set(query):
        raise ValueError('Query labels forbidden')
    frame = pd.concat([support.drop(columns=['soh', 'capacity']), query], ignore_index=True)
    x, valid = design(support, frame, np.r_[raw_support, raw_query], record['pca'], record['view'])
    x = torch.tensor(np.clip(np.nan_to_num(x, nan=0.0) / np.array(record['normalization']['scale']), -10, 10), dtype=torch.float32, device='cuda')
    head = Head().to('cuda').eval()
    head.load_state_dict(record['state'], strict=True)
    with torch.no_grad():
        delta = head(x)[20:].cpu().numpy()
    return np.asarray(raw_query) + np.where(valid[20:], delta, 0.0)
