"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import pandas as pd
from models.domain import qv_head as qh
import torch as torch

def eligible(s, counts):
    if 'qv_valid' not in s or 'experiment_temperature_C' not in s:
        return False
    if abs(float(s.cc_current_A.median()) / 3.5 - 0.5) > 0.15:
        return False
    ts = s.experiment_temperature_C.dropna().unique()
    return len(ts) == 1 and counts.get(f'{float(ts[0]):g}|0.5', 0) >= 4 and bool(s.qv_valid.all())

def design(s, q, raw, view):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    x, valid = qh.design(s, q, raw, {}, 'coarse')
    cols = ['CC Q', 'CV Q', 'CC charge time']
    ref = s[cols].median().to_numpy(float)
    high = (q[cols].to_numpy(float) - ref) / np.maximum(abs(ref), 1e-06)
    total = float(s['CC Q'].median() + s['CV Q'].median())
    amplitude = (q['CC Q'].to_numpy() + q['CV Q'].to_numpy() - total) / max(abs(total), 1e-06)
    control = np.column_stack([high, amplitude, x[:, 4:]])
    valid &= np.isfinite(control).all(axis=1)
    return (control if view == 'high' else x, valid)

def inference(record, s, q, raw):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    frame = pd.concat([s.drop(columns=['soh', 'capacity']), q], ignore_index=True)
    x, valid = design(s, frame, np.r_[s.soh.to_numpy(float), raw], record['view'])
    x = torch.tensor(np.clip(np.nan_to_num(x, nan=0.0) / np.array(record['normalization']['scale']), -10, 10), dtype=torch.float32, device='cuda')
    head = qh.Head().to('cuda').eval()
    head.load_state_dict(record['state'], strict=True)
    with torch.no_grad():
        delta = head(x)[20:].cpu().numpy()
    return np.asarray(raw) + np.where(valid[20:], delta, 0.0)

def project_branches(s, old_mixed_raw, pure_raw, corrected_mixed_raw):
    delta = np.asarray(corrected_mixed_raw) - np.asarray(old_mixed_raw)
    curve = 2 * np.asarray(old_mixed_raw) - np.asarray(pure_raw)

    def project(x):
        return np.clip(np.minimum.accumulate(np.r_[float(s.soh.iloc[-1]), x])[1:], 0, 1.1)
    return 0.5 * (project(np.asarray(pure_raw) + delta) + project(curve + delta))
