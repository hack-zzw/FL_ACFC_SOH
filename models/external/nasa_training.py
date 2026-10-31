"""Early-label-only NASA client training; no complete target trajectory fitting."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from models.external.initialization import initialized_model
from models.external.nasa_readout import FEATURES, RAW_INDICES
from models.external.adaptation import input_features
from models.external.nasa_support import check_features
from models.external.nasa_features import head_with_alpha

def validate_support(clients, limit):
    if not clients:
        raise ValueError('No training clients')
    for client, frame in clients.items():
        if frame.empty or frame.record_index.min() < 0 or frame.record_index.max() >= limit:
            raise ValueError(f'Future observation entered training: {client}')
        if frame.duplicated(['battery_id', 'record_index']).any():
            raise ValueError('Duplicate support row')
        if not np.isfinite(frame.soh.to_numpy(float)).all():
            raise ValueError('Nonfinite support label')
        for _, g in frame.groupby('battery_id'):
            if not np.array_equal(np.sort(g.record_index.to_numpy()), np.arange(len(g))):
                raise ValueError('Training must be an initial chronological prefix')

def client_weights(frame):
    weights = np.empty(len(frame))
    for _, pos in frame.groupby('battery_id').indices.items():
        weights[pos] = 1.0 / (frame.battery_id.nunique() * len(pos))
    return weights

class EarlyModel(nn.Module):

    def __init__(self, backbone, variant, encoder_center, encoder_scale, raw_center, raw_scale, rep_center, rep_scale, prior):
        super().__init__()
        self.backbone, self.variant = (backbone, variant)
        for name, value in (('encoder_center', encoder_center), ('encoder_scale', encoder_scale), ('raw_center', raw_center), ('raw_scale', raw_scale), ('rep_center', rep_center), ('rep_scale', rep_scale)):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        self.prior = nn.Parameter(torch.tensor(prior, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def representation(self, enc, raw):
        embedding = self.backbone.solution.get_embedding(enc.float()).double()
        return torch.cat((embedding, raw.double()), 1) if self.variant == 'dual' else embedding

    def forward(self, enc, raw):
        return (self.representation(enc, raw) - self.rep_center) / self.rep_scale

def tensors(model, features, device):
    check_features(features)
    values = features[FEATURES].to_numpy(float)
    enc = (values - model.encoder_center.cpu().numpy()) / model.encoder_scale.cpu().numpy()
    enc[:, [i for i in range(len(FEATURES)) if i not in RAW_INDICES]] = 0
    raw = (values[:, RAW_INDICES] - model.raw_center.cpu().numpy()) / model.raw_scale.cpu().numpy()
    if not np.isfinite(enc).all() or not np.isfinite(raw).all():
        raise ValueError('Nonfinite features')
    return (torch.tensor(np.clip(enc, -10, 10), device=device, dtype=torch.float32), torch.tensor(np.clip(raw, -10, 10), device=device, dtype=torch.float64))

def embedded(model, features, device):
    return model(*tensors(model, features, device))

def initialize(config, source_state, seed, family, variant, clients, scaler, device, limit, prior_strength):
    """Aggregate client sufficient statistics, never pool raw labels for fitting."""
    validate_support(clients, limit)
    backbone = initialized_model(config, source_state if family == 'source' else None, device, seed)
    backbone.eval()
    summaries = []
    for c, g in sorted(clients.items()):
        x = g[FEATURES].to_numpy(float)[:, RAW_INDICES]
        w = client_weights(g)
        summaries.append((w @ x, w @ (x * x)))
    raw_center = np.mean([m for m, _ in summaries], axis=0)
    raw_scale = np.sqrt(np.maximum(np.mean([s for _, s in summaries], axis=0) - raw_center ** 2, 0)).clip(1e-06)
    dim = 36 + (len(RAW_INDICES) if variant == 'dual' else 0)
    model = EarlyModel(backbone, variant, scaler.全局.均值.astype(float), scaler.全局.标准差.astype(float), raw_center, raw_scale, np.zeros(dim), np.ones(dim), np.zeros(dim)).to(device)
    summaries, local_features = ([], {})
    with torch.no_grad():
        for c, g in sorted(clients.items()):
            h = model.representation(*tensors(model, input_features(g), device)).cpu().numpy()
            local_features[c] = h
            w = client_weights(g)
            summaries.append((w @ h, w @ (h * h)))
    center = np.mean([m for m, _ in summaries], axis=0)
    scale = np.sqrt(np.maximum(np.mean([s for _, s in summaries], axis=0) - center ** 2, 0)).clip(0.1)
    normals, rhs = ([], [])
    for c, g in sorted(clients.items()):
        x = (local_features[c] - center) / scale
        y = g.soh.to_numpy(float).copy()
        for _, pos in g.groupby('battery_id').indices.items():
            x[pos] -= x[pos].mean(0)
            y[pos] -= y[pos].mean()
        w = client_weights(g)
        normals.append(x.T @ (w[:, None] * x))
        rhs.append(x.T @ (w * y))
    normal, target = (np.mean(normals, axis=0), np.mean(rhs, axis=0))
    prior = np.linalg.solve(normal + prior_strength * np.eye(dim), target)
    with torch.no_grad():
        model.rep_center.copy_(torch.tensor(center, device=device))
        model.rep_scale.copy_(torch.tensor(scale, device=device))
        model.prior.copy_(torch.tensor(prior, device=device))
    return model

def active(name, scope):
    if name == 'prior':
        return True
    return scope == 'last' and name.startswith('backbone.solution.encoder.') and any((s in name for s in ('.投影.', '.单元.1.', '.注意力.1.')))

def train(model, clients, candidate, device, limit):
    """FedAvg of centered supervised client updates; offsets stay local."""
    validate_support(clients, limit)
    model.eval()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        p.requires_grad = active(n, candidate['scope'])
    shared = {n: v.detach().clone() for n, v in model.state_dict().items()}
    batches = {c: (tensors(model, input_features(g), device), torch.tensor(g.soh.to_numpy(float), device=device), list(g.groupby('battery_id').indices.values())) for c, g in clients.items()}
    logs = []
    for r in range(candidate['rounds']):
        updates = []
        for c in sorted(clients):
            model.load_state_dict(shared)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)
            (enc, raw), y, positions = batches[c]
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                pred = model(enc, raw) @ model.prior
                losses = []
                for pos in positions:
                    pp, yy = (pred[pos], y[pos])
                    error = pp - pp.mean() - (yy - yy.mean())
                    losses.append(error.square().mean())
                data_loss = torch.stack(losses).mean()
                anchor = torch.stack([(p - initial[n]).square().mean() for n, p in model.named_parameters() if p.requires_grad]).mean()
                loss = data_loss + 0.01 * anchor
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                logs.append(dict(client=c, round=r + 1, step=step, data_loss=float(data_loss.detach()), anchor=float(anchor.detach())))
            updates.append({n: v.detach().clone() for n, v in model.state_dict().items() if active(n, candidate['scope'])})
        for n in updates[0]:
            shared[n] = torch.stack([u[n] for u in updates]).mean(0)
    model.load_state_dict(shared)
    frozen = sum((not active(n, candidate['scope']) for n in initial))
    for n, p in model.named_parameters():
        if not active(n, candidate['scope']) and (not torch.equal(p, initial[n])):
            raise AssertionError(f'Frozen parameter changed: {n}')
    return (logs, frozen)

def predict(model, support, features, head, device):
    check_features(features)
    model.eval()
    with torch.no_grad():
        sx = embedded(model, input_features(support), device)
        qx = embedded(model, features, device)
        pred, fit = head_with_alpha(sx, qx, torch.tensor(support.soh.to_numpy(float), device=device), support.battery_id.astype(str).tolist(), features.battery_id.astype(str).tolist(), model.prior, model.bias, head['strength'], head['alpha'])
    return (pred.cpu().numpy(), fit)

def payload(model):
    return dict(variant=model.variant, state={n: v.detach().cpu().clone() for n, v in model.state_dict().items()})

def restore(saved, config, device, seed):
    s = saved['state']
    model = EarlyModel(initialized_model(config, None, device, seed), saved['variant'], *[s[n].numpy() for n in ('encoder_center', 'encoder_scale', 'raw_center', 'raw_scale', 'rep_center', 'rep_scale', 'prior')]).to(device)
    model.load_state_dict(s, strict=True)
    model.eval()
    return model
