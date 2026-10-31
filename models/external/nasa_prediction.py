"""Support-only personalization and causal delayed NASA client joining."""
from __future__ import annotations
import numpy as np
from models.external.nasa_training import predict
from models.external.adaptation import input_features
JOINER = 'NASA_09'
OPTIONS = ('unchanged', 'trend_1', 'trend_10', 'personal_head', 'personal_last')

def trend_predict(model, support, features, head, device, strength):
    """Battery-specific ridge residual slope; fits observed support labels only."""
    base, _ = predict(model, support, features, head, device)
    fitted, _ = predict(model, support, input_features(support), head, device)
    coefficients = {}
    correction = np.zeros(len(features))
    for bid, pos in support.groupby('battery_id').indices.items():
        x = support.iloc[pos].cycle_index.to_numpy(float)
        center, scale = (float(x.mean()), max(float(x.std()), 1.0))
        z = (x - center) / scale
        residual = support.iloc[pos].soh.to_numpy(float) - fitted[pos]
        residual -= residual.mean()
        slope = float(np.mean(z * residual) / (np.mean(z * z) + strength))
        coefficients[str(bid)] = dict(center=center, scale=scale, slope=slope)
        ix = np.flatnonzero(features.battery_id.to_numpy() == bid)
        correction[ix] = slope * (features.iloc[ix].cycle_index.to_numpy(float) - center) / scale
    return (base + correction, coefficients)

def option_predict(model, support, features, head, option, device):
    if option.startswith('trend_'):
        return trend_predict(model, support, features, head, device, float(option.split('_')[1]))
    pred, fit = predict(model, support, features, head, device)
    return (pred, fit)
