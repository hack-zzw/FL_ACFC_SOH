"""Final domain inference: conditional heads, support calibration, residuals and coverage routing."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain.heads import Head as Head
from models.domain.nca_rate import Residual as Residual
from dataloader.conditional_features import design as design
from dataloader.history_features import history_features as history_features
import numpy as np
import pandas as pd
from models.domain.projection import project as project
from utils.state import restore_scaler as restore_scaler
import torch as torch
from configs.schema import 实验配置 as 实验配置
from models.encoder import 按配置创建模型 as 按配置创建模型
from models.domain.mit_calibration import apply_quality as apply_quality
import copy as copy
from models.domain import heads as producer
from models.domain import qv_head as qh
from models.domain import nca_half_rate as half
from models.domain import validation_masks as deployed
from models.domain import routing as scope
from models.domain.temperature_transport import transport as transport
from models.domain import context as c
from dataloader import history16 as history
from models.domain import xjtu_head as run
from models.domain.temperature_masks import masks as masks
from models.domain.temperature_weights import weight as weight

def conditional_prediction(bundle, domain, support, query_features, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    if len(support) != 20:
        raise ValueError('Exactly20 support rows required')
    if len(query_features) != len(base_raw) or len(base_raw) != len(base_final):
        raise ValueError('Query/prediction mismatch')
    record = bundle['conditional_heads'][domain]
    cfg = 实验配置(**bundle['config'])
    scaler = restore_scaler(bundle['scaler'])
    observed = pd.concat([support.drop(columns=['soh', 'capacity', 'true_soh'], errors='ignore'), query_features], ignore_index=True)
    model = 按配置创建模型(cfg).to(device).eval()
    model.load_state_dict(bundle['domain_states'][domain], strict=True)
    with torch.no_grad():
        embedded = model.solution.get_embedding(torch.from_numpy(scaler.变换(observed)).to(device)).cpu().numpy().astype(float)
    physical = design(support, observed, domain, 'support_conditioned_window_shape')
    x = np.column_stack([embedded, physical])
    if record.get('history', False):
        h = history_features(observed)
        columns = [c for c in h if c.startswith('history_')]
        keys = pd.MultiIndex.from_frame(observed[['battery_id', 'cycle_index']])
        extra = h.set_index(['battery_id', 'cycle_index']).loc[keys, columns].to_numpy(float)
        x = np.column_stack([x, extra])
    missing = ~np.isfinite(x)
    mean = np.array(record['normalization']['mean'])
    sd = np.array(record['normalization']['sd'])
    scaled = np.clip((np.where(missing, mean, x) - mean) / sd, -10, 10)
    tensor = torch.from_numpy(np.column_stack([scaled, missing.astype(float)]).astype(np.float32)).to(device)
    head = Head(record['dimension'], record['width']).to(device).eval()
    head.load_state_dict(record['state'], strict=True)
    with torch.no_grad():
        v = head(tensor)
    extra_record = record.get('rate_residual')
    if extra_record:
        observed_rate = float(support.cc_current_A.median()) / float(extra_record['nominal_Ah'])
        centers = np.array(extra_record['centers'])
        group = int(np.argmin(abs(centers - observed_rate)))
        if not np.isfinite(observed_rate) or abs(centers[group] - observed_rate) > extra_record['rate_tolerance']:
            group = -1
        counts = extra_record.get('training_condition_counts')
        if counts is not None:
            temps = support.get('experiment_temperature_C', pd.Series(dtype=float)).dropna().unique()
            if len(temps) != 1:
                group = -1
            elif group >= 0:
                key = f'{float(temps[0]):g}|{float(centers[group]):g}'
                if counts.get(key, 0) < extra_record['minimum_training_batteries']:
                    group = -1
        if group >= 0:
            residual = Residual(extra_record['dimension']).to(device).eval()
            residual.load_state_dict(extra_record['state'], strict=True)
            with torch.no_grad():
                v = v + residual(tensor)[:, group]
    v = v.cpu().numpy()
    extra_raw = float(np.median(support.soh)) + v[20:] - np.median(v[:20])
    extra_final = np.clip(extra_raw, 0, 1.1) if domain == 'NCM|XJTU' else project(extra_raw, float(support.soh.iloc[-1]))
    weight = float(record['blend'])
    assert weight == 0.5
    return (0.5 * (np.asarray(base_raw, float) + extra_raw), 0.5 * (np.asarray(base_final, float) + extra_final))

def supervised_prediction(bundle, domain, support, query_features, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    work = bundle
    if domain == 'NCA|TJU':
        record = bundle['conditional_heads'][domain]
        residual = record.get('rate_residual', {})
        if 'private_1C' in residual:
            observed = float(support.cc_current_A.median()) / float(residual['nominal_Ah'])
            if np.isfinite(observed) and abs(observed - 1.0) <= residual['rate_tolerance']:
                work = copy.copy(bundle)
                work['conditional_heads'] = copy.copy(bundle['conditional_heads'])
                work['conditional_heads'][domain] = copy.copy(record)
                work['conditional_heads'][domain]['rate_residual'] = copy.copy(residual)
                work['conditional_heads'][domain]['rate_residual']['state'] = residual['private_1C']['state']
    return conditional_prediction(work, domain, support, query_features, base_raw, base_final, device=device)

def correct_mit(bundle, support, query_features, mixed_raw, current_final):
    return apply_quality(bundle, support, query_features, mixed_raw, current_final)

def eligible_nca(record, support):
    residual = record.get('rate_residual', {})
    if 'private_1C' not in residual:
        return False
    rate = float(support.cc_current_A.median()) / float(residual['nominal_Ah'])
    if not np.isfinite(rate) or abs(rate - 1.0) > residual['rate_tolerance']:
        return False
    temperatures = support.get('experiment_temperature_C', pd.Series(dtype=float)).dropna().unique()
    if len(temperatures) != 1:
        return False
    counts = residual.get('training_condition_counts', {})
    return counts.get(f'{float(temperatures[0]):g}|1', 0) >= residual['minimum_training_batteries']

def residual_prediction(bundle, domain, support, query_features, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    raw, final = supervised_prediction(bundle, domain, support, query_features, base_raw, base_final, device=device)
    record = bundle['conditional_heads'][domain]
    if domain == 'NCA|TJU' and record.get('one_C_fusion') == 'private_branch_only' and eligible_nca(record, support):
        raw = 2 * raw - np.asarray(base_raw, float)
        final = 2 * final - np.asarray(base_final, float)
    if domain == 'NCM|XJTU' and 'final_output_residual' in record:
        from models.domain.xjtu_residual import apply_correction
        final = apply_correction(record['final_output_residual'], support, query_features, raw, final, np.asarray(base_raw, float), np.asarray(base_final, float))
    return (raw, final)

def qv_prediction(bundle, domain, support, query_features, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    raw, final = residual_prediction(bundle, domain, support, query_features, base_raw, base_final, device=device)
    record = bundle['conditional_heads'][domain]
    extra = record.get('coarse_qv_residual')
    if domain != 'NCA|TJU' or extra is None or (not eligible_nca(record, support)):
        return (raw, final)
    if 'qv_valid' not in support or not bool(support.qv_valid.all()):
        return (raw, final)
    if 'qv_valid' not in query_features or not bool(query_features.qv_valid.any()):
        return (raw, final)
    cfg = 实验配置(**bundle['config'])
    scaler = restore_scaler(bundle['scaler'])
    encoder = 按配置创建模型(cfg).to(device).eval()
    encoder.load_state_dict(bundle['domain_states'][domain], strict=True)
    observed = pd.concat([support, query_features.assign(soh=0.0, capacity=0.0)], ignore_index=True)
    tasks = producer.build(observed, encoder, scaler)
    producer.normalize(tasks, record['normalization'])
    assert len(tasks) == 1
    head = producer.Head(record['dimension'], record['width']).to(device).eval()
    head.load_state_dict(record['state'], strict=True)
    private = qh.teachers.rate.Residual(record['dimension']).to(device).eval()
    private.load_state_dict(record['rate_residual']['private_1C']['state'], strict=True)
    with torch.no_grad():
        v = (head(tasks[0]['tensor']) + private(tasks[0]['tensor'])[:, 2]).cpu().numpy()
    support_raw = float(np.median(support.soh)) + v[:20] - float(np.median(v[:20]))
    corrected = qh.inference(extra, support, query_features, support_raw, raw)
    projected = np.clip(np.minimum.accumulate(np.r_[float(support.soh.iloc[-1]), corrected])[1:], 0, 1.1)
    return (corrected, projected)

def half_rate_prediction(bundle, domain, support, query_features, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(query_features):
        raise ValueError('Query labels forbidden')
    raw, final = qv_prediction(bundle, domain, support, query_features, base_raw, base_final, device=device)
    extra = bundle['conditional_heads'][domain].get('half_C_qv_residual')
    if domain != 'NCA|TJU' or extra is None or (not half.eligible(support, extra['training_condition_counts'])):
        return (raw, final)
    corrected = half.inference(extra, support, query_features, raw)
    projected = half.project_branches(support, raw, np.asarray(base_raw), corrected)
    return (corrected, projected)
DOMAIN = 'NCM|XJTU'
_ENCODERS = {}
_HEADS = {}
_OWNERS = []

def xjtu_head_values(bundle, record, s, q, device='cuda'):
    key = (id(bundle), device)
    if key not in _ENCODERS:
        cfg = c.read_config(c.OLD / 'config.json')
        encoder = c.按配置创建模型(cfg).to(device).eval()
        encoder.load_state_dict(bundle['domain_states'][DOMAIN], strict=True)
        for p in encoder.parameters():
            p.requires_grad_(False)
        _ENCODERS[key] = encoder
        _OWNERS.append(bundle)
    frame = pd.concat([s, q.assign(soh=0.0, capacity=0.0)], ignore_index=True)
    tasks = run.curve.build(frame, _ENCODERS[key], c.restore_scaler(bundle['scaler']))
    run.curve.normalize(tasks, record['normalization'])
    run.expanded.append_low(tasks, frame, record['low_voltage_input']['scale'])
    if record['dimension'] == 238:
        tasks = history.augment(tasks, frame, record['history_normalization'], real=True)
    hkey = (id(record), device)
    if hkey not in _HEADS:
        _HEADS[hkey] = run.coverage.head_from_record(record)
        _OWNERS.append(record)
    assert len(tasks) == 1
    with torch.no_grad():
        v = _HEADS[hkey](tasks[0]['tensor']).cpu().numpy()
    raw = float(np.median(s.soh.to_numpy(float))) + v[20:] - float(np.median(v[:20]))
    return (raw, np.clip(raw, 0, 1.1))

def protected_prediction(bundle, domain, s, q, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    oldraw, oldfinal = half_rate_prediction(bundle, domain, s, q, base_raw, base_final, device=device)
    extra = bundle.get('xjtu_confirmation')
    if domain != DOMAIN or extra is None:
        return (oldraw, oldfinal)
    G, H, _ = scope.masks(extra['coverage'], s, q)
    valid, _ = deployed.mask(s, q)
    G &= valid
    raw = np.asarray(oldraw).copy()
    final = np.asarray(oldfinal).copy()
    for mask, record in [(G, extra['ordinary']), (H, extra['protected'])]:
        if not mask.any():
            continue
        ss = s
        qq = q
        if record is extra['protected']:
            observed = pd.concat([s, q.assign(soh=0.0, capacity=0.0)], ignore_index=True)
            shifted, _ = transport(observed, extra['coverage'])
            ss = shifted.iloc[:20]
            qq = shifted.iloc[20:].drop(columns=['soh', 'capacity'])
        r, f = xjtu_head_values(bundle, record, ss, qq, device=device)
        raw[mask] = 0.5 * (np.asarray(base_raw)[mask] + r[mask])
        final[mask] = 0.5 * (np.asarray(base_final)[mask] + f[mask])
    return (raw, final)

def cooling_prediction(bundle, domain, s, q, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    if domain != DOMAIN or not bundle.get('xjtu_cooling'):
        return protected_prediction(bundle, domain, s, q, base_raw, base_final, device=device)
    raw, final = half_rate_prediction(bundle, domain, s, q, base_raw, base_final, device=device)
    record = bundle['xjtu_confirmation']
    G, H, _ = masks(record['coverage'], s, q)
    from models.domain.validation_masks import mask
    good, _ = mask(s, q)
    G &= good
    raw = np.asarray(raw).copy()
    final = np.asarray(final).copy()
    for use, head in [(G, record['ordinary']), (H, record['protected'])]:
        if not use.any():
            continue
        ss = s
        qq = q
        if head is record['protected']:
            observed = pd.concat([s, q.assign(soh=0.0, capacity=0.0)], ignore_index=True)
            shifted, _ = transport(observed, record['coverage'])
            ss = shifted.iloc[:20]
            qq = shifted.iloc[20:].drop(columns=['soh', 'capacity'])
        r, f = xjtu_head_values(bundle, head, ss, qq, device=device)
        raw[use] = 0.5 * (np.asarray(base_raw)[use] + r[use])
        final[use] = 0.5 * (np.asarray(base_final)[use] + f[use])
    return (raw, final)

def blend_curve_head(bundle, domain, s, q, base_raw, base_final, device='cuda'):
    if {'soh', 'capacity', 'true_soh'} & set(q):
        raise ValueError('Query labels forbidden')
    H = protected_prediction(bundle, domain, s, q, base_raw, base_final, device=device)
    extra = bundle.get('xjtu_smooth')
    if domain != DOMAIN or extra is None:
        return H
    a = weight(bundle['xjtu_confirmation']['coverage'], extra['source_temperature_sd'], s, q)
    if not a.any():
        return H
    G = cooling_prediction(bundle, domain, s, q, base_raw, base_final, device=device)
    return tuple((np.asarray(h) + a * (np.asarray(g) - np.asarray(h)) for g, h in zip(G, H)))
