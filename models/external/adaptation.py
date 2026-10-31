"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
from __future__ import annotations
import numpy as np
import torch
from configs.schema import 特征列
from models.external.initialization import initialized_model
METHODS = ('bias', 'affine', 'embedding_residual', 'feature_residual', 'head', 'partial')

def input_features(frame):
    return frame.drop(columns=['soh', 'capacity'], errors='ignore')

class SupportTransform:
    """Only support observations may trigger unreliable-moment suppression."""

    def __init__(self, scaler, support, mode):
        self.scaler, self.mode = (scaler, mode)
        self.mask = []
        if mode == 'support_guard' and (not support.empty):
            z = scaler.变换(input_features(support))
            if float(np.median(support['current std'])) < 0.01:
                for column in ('current kurtosis', 'current skewness'):
                    index = 特征列.index(column)
                    if float(np.median(np.abs(z[:, index]))) > 5:
                        self.mask.append(index)

    def apply(self, frame):
        if 'soh' in frame or 'capacity' in frame:
            raise ValueError('Transform and prediction must not receive query labels')
        z = self.scaler.变换(frame)
        if self.mode in {'clip5', 'support_guard'}:
            z = np.clip(z, -5, 5)
        elif self.mode == 'clip10':
            z = np.clip(z, -10, 10)
        elif self.mode != 'raw':
            raise ValueError(self.mode)
        if self.mask:
            z[:, self.mask] = 0
        return z.astype(np.float32)

def infer(model, x, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(device)
        embedding = model.solution.get_embedding(tensor)
        output = model.solution.predictor(embedding)
        if model.solution.protocol_adapter is not None:
            output += model.solution.protocol_adapter(embedding)
        return (output.cpu().numpy().reshape(-1).astype(float), embedding.cpu().numpy().astype(float))

def ridge_residual(train_x, test_x, residual, strength):
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.maximum(scale, 0.1)
    x = np.clip((train_x - center) / scale, -10, 10)
    q = np.clip((test_x - center) / scale, -10, 10)
    r_center = float(residual.mean())
    matrix = x.T @ x / len(x) + float(strength) * np.eye(x.shape[1])
    weights = np.linalg.solve(matrix, x.T @ (residual - r_center) / len(x))
    return (q @ weights + r_center, {'center': center.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(), 'intercept': r_center})

def fine_tune(model, x, y, device, candidate):
    mode = candidate['method']
    prefixes = ['solution.predictor.']
    if mode == 'partial':
        prefixes += ['solution.encoder.投影.', 'solution.encoder.单元.1.', 'solution.encoder.注意力.1.']
    parameters, reference = ([], [])
    for name, parameter in model.named_parameters():
        active = any((name.startswith(prefix) for prefix in prefixes))
        parameter.requires_grad = active
        if active:
            parameters.append(parameter)
            reference.append(parameter.detach().clone())
    optimizer = torch.optim.AdamW(parameters, lr=candidate['lr'], weight_decay=0.0001)
    features = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y.astype(np.float32)).to(device).reshape(-1, 1)
    model.eval()
    final_loss = None
    for _ in range(candidate['epochs']):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = torch.nn.functional.huber_loss(prediction, labels, delta=0.05)
        penalty = sum(((parameter - initial).square().sum() for parameter, initial in zip(parameters, reference)))
        loss = loss + candidate.get('l2sp', 0.0) * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 2.0)
        optimizer.step()
        final_loss = float(loss.detach())
    return {'updated_parameters': sum((p.numel() for p in parameters)), 'final_loss': final_loss}

def adapt_predict(payload, config, support, query, scaler, device, candidate, cached=None):
    if 'soh' in query or 'capacity' in query:
        raise ValueError('Query labels may not enter adaptation or inference')
    transform = SupportTransform(scaler, support, candidate['transform'])
    sx, qx = (transform.apply(input_features(support)), transform.apply(query))
    model = initialized_model(config, payload['global_state'], device, payload['seed'])
    ys = support.soh.to_numpy(float)
    if cached is None:
        ps, hs = infer(model, sx, device)
        pq, hq = infer(model, qx, device)
    else:
        ps, hs, pq, hq = cached
    method = candidate['method']
    state = {'masked_features': [特征列[i] for i in transform.mask], 'candidate': candidate}
    if method == 'bias':
        bias = float((ys - ps).mean())
        prediction = pq + bias
        state.update(bias=bias)
    elif method == 'affine':
        p_center, y_center = (float(ps.mean()), float(ys.mean()))
        centered = ps - p_center
        variance = float(np.mean(centered ** 2))
        strength = candidate['strength']
        slope = (float(np.mean(centered * (ys - y_center))) + strength) / (variance + strength)
        slope = float(np.clip(slope, 0.0, 3.0))
        bias = y_center - slope * p_center
        prediction = slope * pq + bias
        state.update(slope=slope, bias=bias)
    elif method in {'embedding_residual', 'feature_residual'}:
        xs, xq = (hs, hq) if method == 'embedding_residual' else (sx.astype(float), qx.astype(float))
        correction, fit = ridge_residual(xs, xq, ys - ps, candidate['strength'])
        prediction = pq + correction
        state.update(fit=fit)
    elif method in {'head', 'partial'}:
        state.update(fine_tune(model, sx, ys, device, candidate))
        prediction, _ = infer(model, qx, device)
    else:
        raise ValueError(method)
    return (prediction, state, model)


def simple_baseline(support, query, mode):
    if 'soh' in query or 'capacity' in query:
        raise ValueError('Baseline prediction may not receive query labels')
    prediction = np.empty(len(query))
    fits = {}
    for battery_id, positions in query.groupby('battery_id', sort=True).indices.items():
        s = support[support.battery_id == battery_id].sort_values('cycle_index')
        t, y = (s.cycle_index.to_numpy(float), s.soh.to_numpy(float))
        q = query.iloc[positions].cycle_index.to_numpy(float)
        if mode == 'anchor':
            prediction[positions] = y[-1]
        else:
            centered = t - t.mean()
            denominator = float(np.dot(centered, centered))
            slope = float(np.dot(centered, y - y.mean()) / denominator) if denominator else 0.0
            prediction[positions] = y.mean() + slope * (q - t.mean())
            fits[battery_id] = {'slope': slope, 'center': float(t.mean()), 'mean_soh': float(y.mean())}
    return (prediction, fits)
