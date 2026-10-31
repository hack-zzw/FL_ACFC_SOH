"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
import numpy as np
import pandas as pd
import torch
from models.external.source import PROJECT, load_scaler, read_config
from models.external.initialization import initialized_model
from models.external.adaptation import METHODS, SupportTransform, infer

def predict_saved_client(checkpoint, features, source_artifacts, device):
    if 'soh' in features or 'capacity' in features:
        raise ValueError('Prediction input must contain features only, without SOH or capacity')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    method = payload['metadata']['method']
    if method not in METHODS:
        raise ValueError('This entry is for persisted ACfC adaptation methods')
    config = read_config(PROJECT / 'configs/best_no_pinn_experiment_config.json')
    robust_path = source_artifacts / 'robust_scaler.json'
    if robust_path.exists():
        import json
        from models.external.scaler import RobustSourceScaler
        scaler = RobustSourceScaler(json.loads(robust_path.read_text(encoding='utf-8')))
    else:
        scaler = load_scaler(source_artifacts / 'source_scaler.json')
    adaptation = payload['adaptation']
    candidate = adaptation['candidate']
    transform = SupportTransform(scaler, pd.DataFrame(), candidate['transform'])
    from configs.schema import 特征列
    transform.mask = [特征列.index(column) for column in adaptation['masked_features']]
    x = transform.apply(features)
    model = initialized_model(config, payload['state'], device, payload['metadata']['seed'])
    prediction, embedding = infer(model, x, device)
    if method == 'bias':
        prediction += adaptation['bias']
    elif method == 'affine':
        prediction = adaptation['slope'] * prediction + adaptation['bias']
    elif method in {'embedding_residual', 'feature_residual'}:
        fit = adaptation['fit']
        values = embedding if method == 'embedding_residual' else x.astype(float)
        standardized = np.clip((values - np.array(fit['center'])) / np.array(fit['scale']), -10, 10)
        prediction += standardized @ np.array(fit['weights']) + fit['intercept']
    return prediction
