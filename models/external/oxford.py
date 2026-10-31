"""NASA protocol holdout and Oxford battery holdout development optimization."""
from __future__ import annotations
import numpy as np
from configs.schema import 特征列
from models.external.initialization import initialized_model
from models.external.adaptation import METHODS, SupportTransform, adapt_predict, infer, input_features, ridge_residual, simple_baseline
SPLIT_SEED = 20261007
SHOTS = (5, 10, 20)
METRICS = ['mae', 'mape', 'rmse', 'r2']
DATASETS = ('NASA', 'Oxford')
ACFC_METHODS = METHODS + ('nasa_residual',)

def nasa_ridge_fit(support, query, sx, qx, target, strength, per_battery):
    offsets = {}
    if per_battery:
        sx, qx, target = (sx.copy(), qx.copy(), target.copy())
        for battery, positions in support.groupby('battery_id', sort=True).indices.items():
            center, offset = (sx[positions].mean(axis=0), float(target[positions].mean()))
            offsets[str(battery)] = {'center': center.tolist(), 'offset': offset}
            sx[positions] -= center
            target[positions] -= offset
        for battery, positions in query.groupby('battery_id', sort=True).indices.items():
            qx[positions] -= np.asarray(offsets[str(battery)]['center'])
    prediction, fit = ridge_residual(sx, qx, target, strength)
    fit['battery_offsets'] = offsets
    if per_battery:
        for battery, positions in query.groupby('battery_id', sort=True).indices.items():
            prediction[positions] += offsets[str(battery)]['offset']
    return (prediction, fit)

def nasa_ridge_predict(features, x, fit):
    x = x.astype(float).copy()
    offsets = fit.get('battery_offsets', {})
    if offsets:
        for battery, positions in features.groupby('battery_id', sort=True).indices.items():
            if str(battery) not in offsets:
                raise ValueError('Battery has no support calibration')
            x[positions] -= np.asarray(offsets[str(battery)]['center'])
    z = np.clip((x - np.asarray(fit['center'])) / np.asarray(fit['scale']), -10, 10)
    prediction = z @ np.asarray(fit['weights']) + fit['intercept']
    if offsets:
        for battery, positions in features.groupby('battery_id', sort=True).indices.items():
            prediction[positions] += offsets[str(battery)]['offset']
    return prediction

def fit_predict(payload, config, support, query_features, scaler, device, candidate, cache=None):
    if 'soh' in query_features or 'capacity' in query_features:
        raise ValueError('No query labels permitted')
    method = candidate['method']
    if method == 'nasa_residual':
        transform = SupportTransform(scaler, support, candidate['transform'])
        sx, qx = (transform.apply(input_features(support)), transform.apply(query_features))
        model = initialized_model(config, payload['global_state'], device, payload['seed'])
        if cache is None:
            ps, _ = infer(model, sx, device)
            pq, _ = infer(model, qx, device)
        else:
            ps, _, pq, _ = cache
        weight = candidate['prior_weight']
        correction, fit = nasa_ridge_fit(support, query_features, sx.astype(float), qx.astype(float), support.soh.to_numpy(float) - weight * ps, candidate['strength'], candidate['per_battery'])
        return (weight * pq + correction, {'candidate': candidate, 'fit': fit}, model)
    if method in METHODS:
        return adapt_predict(payload, config, support, query_features, scaler, device, candidate, cache)
    if method in {'anchor', 'linear_trend'}:
        prediction, state = simple_baseline(support, query_features, method)
        return (prediction, {'candidate': candidate, 'fit': state}, None)
    transform = SupportTransform(scaler, support, candidate['transform'])
    sx, qx = (transform.apply(input_features(support)), transform.apply(query_features))
    model = None
    if method == 'standalone_embedding_ridge':
        model = initialized_model(config, payload['global_state'], device, payload['seed'])
        if cache is None:
            _, sx = infer(model, sx, device)
            _, qx = infer(model, qx, device)
        else:
            sx, qx = (cache[1], cache[3])
    elif method != 'standalone_feature_ridge':
        raise ValueError(method)
    prediction, fit = nasa_ridge_fit(support, query_features, sx.astype(float), qx.astype(float), support.soh.to_numpy(float), candidate['strength'], candidate.get('per_battery', False))
    return (prediction, {'candidate': candidate, 'fit': fit, 'masked_features': [特征列[i] for i in transform.mask]}, model)
