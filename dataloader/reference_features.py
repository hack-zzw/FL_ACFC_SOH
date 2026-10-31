"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
from models.domain.ncm_calibration import project as project
COLS = ('CC Q', 'CV Q', 'CC charge time', 'CV charge time')

def design(support, features, domain, view):
    if {'soh', 'capacity'} & set(features):
        raise ValueError('Only support labels may enter the design')
    if len(support) != 20:
        raise ValueError('Exactly20 support labels required')
    q = features[list(COLS)].to_numpy(float)
    s = support[list(COLS)].to_numpy(float)
    if domain == 'LFP|MIT':
        q = np.column_stack((q, q[:, 0] + q[:, 1]))
        s = np.column_stack((s, s[:, 0] + s[:, 1]))
    ref = np.median(s, axis=0)
    if view == 'relative':
        z = np.clip((q - ref) / np.maximum(np.abs(ref), 1e-06), -10.0, 10.0)
    elif view == 'absolute':
        z = q
    else:
        raise ValueError(view)
    y = support.soh.to_numpy(float)
    context = np.array([np.median(y) - 0.95, np.ptp(y), y[-1] - y[0]], float)
    x = np.column_stack([np.ones(len(q)), z, np.tile(context, (len(q), 1))] + [z * c for c in context])
    return (x, float(np.median(y)))

def predict(model, support, query_features):
    x, reference = design(support, query_features, model['domain'], model['view'])
    z = (x - np.array(model['mean'])) / np.array(model['sd'])
    raw = reference + z @ np.array(model['coefficients'])
    return (raw, project(raw, support.soh.iloc[-1]))
