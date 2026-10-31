"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from dataloader.conditional_features import design as design
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
DOMAIN = 'NCM|XJTU'
VIEW = 'support_conditioned_window_shape'

class Head(nn.Module):

    def __init__(self, dimension, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def build(frame, model, scaler, sampled=False):
    tasks = []
    for bid, g in frame.groupby('battery_id', sort=True):
        g = g.sort_values('cycle_index')
        s = g.iloc[:20]
        q = g.iloc[20:]
        if sampled:
            take = np.unique(np.linspace(0, len(q) - 1, min(256, len(q))).astype(int))
            q = q.iloc[take]
        selected = pd.concat([s, q], ignore_index=True)
        physics = design(s, selected.drop(columns=['capacity', 'soh']), DOMAIN, VIEW)
        with torch.no_grad():
            x = torch.from_numpy(scaler.变换(selected)).to('cuda')
            z = model.solution.get_embedding(x).cpu().numpy().astype(float)
        values = np.column_stack([z, physics])
        tasks.append(dict(battery_id=bid, client_id=str(g.client_id.iloc[0]), x=values, y=selected.soh.to_numpy(float) - float(np.median(s.soh)), support_level=float(np.median(s.soh)), support_last=float(s.soh.iloc[-1]), query=q[['cycle_index', 'soh']].copy()))
    return tasks

def normalize(tasks, record=None):
    if record is None:
        values = np.concatenate([t['x'] for t in tasks])
        valid = np.isfinite(values)
        n = valid.sum(axis=0)
        mean = np.where(n > 0, np.where(valid, values, 0).sum(axis=0) / np.maximum(n, 1), 0)
        var = np.where(valid, (values - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
        sd = np.maximum(np.sqrt(var), 1e-05)
        record = dict(mean=mean.tolist(), sd=sd.tolist())
    mean, sd = (np.array(record['mean']), np.array(record['sd']))
    for t in tasks:
        x = t['x']
        missing = ~np.isfinite(x)
        scaled = np.clip((np.where(missing, mean, x) - mean) / sd, -10, 10)
        t['tensor'] = torch.from_numpy(np.column_stack([scaled, missing.astype(float)]).astype(np.float32)).to('cuda')
        t['labels'] = torch.from_numpy(t['y'].astype(np.float32)).to('cuda')
    return record
