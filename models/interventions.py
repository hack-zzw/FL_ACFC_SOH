"""Two fixed interventions in the complete paper predictor;no source refitting."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.paths import ROOT as PACKAGE
PROJECT = PACKAGE
OUT = Path(os.environ.get('SOH_RUN_OUTPUT', str(PACKAGE / 'results/generated'))) / 'ablations'
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = PACKAGE / 'results/reference'
RELEASE = PROJECT
SEEDS = (20260903, 20260904, 20260905)
ARMS = ('full_method', 'uniform_support_budget', 'no_monotonic_projection')
LABELS = {'full_method': '本文方法', 'uniform_support_budget': '统一基础预测头支持适配预算', 'no_monotonic_projection': '去除全部推理单调投影'}
from models.predict import verify, cohort, feature_tables
from configs.schema import 实验配置
from utils.regression import 回归指标
from models import adaptation as adaptation
from models.domain import inference as runtime
from utils.state import restore_rng, restore_scaler
from models.domain.reference_head import apply_reference
CONTEXT = dict(arm='full_method', seed=None, shots=None)
RNG_BANK = {}
SUPPORT_BANK = {}
ADAPTATION_LOG = []
PROJECTION_LOG = defaultdict(lambda: dict(calls=0, disabled_calls=0, elements=0))
ORIGINAL_ADAPT = adaptation.仅训练预测头

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def snapshot_rng():
    s = np.random.get_state()
    return dict(python_rng_state=random.getstate(), numpy_rng_state=(s[0], s[1].tolist(), *s[2:]), torch_rng_state=torch.get_rng_state().clone(), cuda_rng_states=[x.clone() for x in torch.cuda.get_rng_state_all()])

def rng_digest(record):
    return hashlib.sha256(record['torch_rng_state'].numpy().tobytes()).hexdigest()

def adapt_matched(model, support, scaler, cfg, device, epochs):
    seed = CONTEXT['seed']
    shots = len(support)
    bid = str(support.battery_id.iloc[0])
    key = (seed, shots, bid)
    assert shots == CONTEXT['shots'] and support.battery_id.nunique() == 1
    support_hash = hashlib.sha256(pd.util.hash_pandas_object(support[['battery_id', 'cycle_index', 'soh']], index=False).to_numpy().tobytes()).hexdigest()
    if CONTEXT['arm'] == 'full_method':
        assert key not in RNG_BANK
        RNG_BANK[key] = snapshot_rng()
        SUPPORT_BANK[key] = support_hash
    else:
        assert SUPPORT_BANK[key] == support_hash
        restore_rng(RNG_BANK[key])
        assert torch.equal(torch.get_rng_state(), RNG_BANK[key]['torch_rng_state'])
    requested_lr = float(cfg.目标学习率)
    requested_epochs = int(epochs)
    lr, steps = (UNIFORM_LR, UNIFORM_EPOCHS) if CONTEXT['arm'] == 'uniform_support_budget' and shots == 20 else (requested_lr, requested_epochs)
    old = cfg.目标学习率
    cfg.目标学习率 = lr
    encoder_before = {k: v.detach().clone() for k, v in model.solution.encoder.state_dict().items()}
    try:
        result = ORIGINAL_ADAPT(model, support, scaler, cfg, device, steps)
    finally:
        cfg.目标学习率 = old
    assert all((torch.equal(v, model.solution.encoder.state_dict()[k]) for k, v in encoder_before.items()))
    ADAPTATION_LOG.append(dict(arm=CONTEXT['arm'], seed=seed, shots=shots, battery_id=bid, domain=str(support.material.iloc[0]) + '|' + str(support.source.iloc[0]), requested_lr=requested_lr, requested_epochs=requested_epochs, effective_lr=lr, effective_epochs=steps, support_sha256=support_hash, canonical_torch_rng_sha256=rng_digest(RNG_BANK[key]), source_encoder_unchanged=True))
    return result

class MinimumProxy:

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return np.minimum(*args, **kwargs)

    def accumulate(self, values, *args, **kwargs):
        disabled = CONTEXT['arm'] == 'no_monotonic_projection' and CONTEXT['shots'] == 20
        k = (CONTEXT['arm'], self.module)
        row = PROJECTION_LOG[k]
        row['calls'] += 1
        row['elements'] += np.asarray(values).size
        if not disabled:
            return np.minimum.accumulate(values, *args, **kwargs)
        assert not args and (not kwargs)
        row['disabled_calls'] += 1
        return np.asarray(values).copy()

class NumpyView:

    def __init__(self, module):
        self.minimum = MinimumProxy(module)

    def __getattr__(self, name):
        return getattr(np, name)

def projection_sites():
    rows = []
    for module_name in ['models.adaptation', 'models.domain.ncm_calibration', 'models.domain.inference', 'models.domain.nca_half_rate', 'models.domain.projection']:
        p = PROJECT / (module_name.replace('.', '/') + '.py')
        tree = ast.parse(p.read_text(encoding='utf-8'))
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (n.func.attr == 'accumulate') and isinstance(n.func.value, ast.Attribute) and (n.func.value.attr == 'minimum') and isinstance(n.func.value.value, ast.Name) and (n.func.value.value.id == 'np')]
        if nodes:
            name = module_name
            mod = importlib.import_module(name)
            rows.append(dict(module=name, path=str(p), lines=sorted((n.lineno for n in nodes)), sha256=sha(p)))
    assert len(rows) == 5 and sum((len(r['lines']) for r in rows)) == 6
    return rows

def register(sites):
    plan = dict(method='paper_method_20shot_equal3_20261025', arms=list(ARMS), support=20, projection_sites=sites, random_state_matching=True)
    write(OUT / 'registration.json', plan)
    return plan

def infer(bundle, target, features, temperatures, seed):
    cfg = 实验配置(**bundle['config'])
    scaler = restore_scaler(bundle['scaler'])
    restore_rng(bundle)
    for shots in cfg.shots:
        CONTEXT['shots'] = shots
        _, _, rows = adaptation._评估(bundle['method'], bundle['global_state'], bundle['domain_states'], target, scaler, copy.deepcopy(cfg), torch.device('cuda'), seed, shots)
        if shots == 20:
            actual = pd.DataFrame(rows)
            actual['domain'] = actual.material + '|' + actual.source
    CONTEXT['shots'] = 20
    for domain, record in bundle['reference_heads'].items():
        idx = actual.index[actual.domain == domain]
        f = apply_reference(actual.loc[idx], target[target.domain == domain], record)
        actual.loc[idx, ['pred_soh', 'raw_pred_soh']] = f[['pred_soh', 'raw_pred_soh']]
    checks = 0
    if 'ordered_quality_residuals' in bundle:
        held = target[target.domain == 'LFP|MIT'].merge(features['LFP|MIT'], on=['battery_id', 'cycle_index'], validate='one_to_one')
        for bid, g in held.groupby('battery_id', sort=True):
            g = g.sort_values('cycle_index')
            s = g.iloc[:20]
            q = g.iloc[20:].drop(columns=['soh', 'capacity'])
            idx = actual.index[actual.battery_id == bid]
            raw = actual.loc[idx].raw_pred_soh.to_numpy().copy()
            base = actual.loc[idx].pred_soh.to_numpy().copy()
            pred = runtime.correct_mit(bundle, s, q, raw, base)
            actual.loc[idx, 'pred_soh'] = pred
            n = max(1, len(q) // 2)
            np.testing.assert_allclose(runtime.correct_mit(bundle, s, q.iloc[:n], raw[:n], base[:n]), pred[:n], rtol=0, atol=1e-12)
            try:
                runtime.correct_mit(bundle, s, q.assign(true_soh=-999.0), raw, base)
            except ValueError:
                checks += 1
            else:
                raise AssertionError('MIT accepted query labels')
    for domain in bundle['conditional_heads']:
        held = target[target.domain == domain].merge(features[domain], on=['battery_id', 'cycle_index'], validate='one_to_one')
        if domain == 'NCA|TJU':
            held['experiment_temperature_C'] = held.battery_id.map(temperatures)
            held['qv_valid'] = held.qv_valid.fillna(False).astype(bool)
        for bid, g in held.groupby('battery_id', sort=True):
            g = g.sort_values('cycle_index')
            s = g.iloc[:20]
            q = g.iloc[20:].drop(columns=['soh', 'capacity'])
            idx = actual.index[actual.battery_id == bid]
            raw = actual.loc[idx].raw_pred_soh.to_numpy().copy()
            base = actual.loc[idx].pred_soh.to_numpy().copy()
            rr, pp = runtime.blend_curve_head(bundle, domain, s, q, raw, base)
            actual.loc[idx, 'raw_pred_soh'] = rr
            actual.loc[idx, 'pred_soh'] = pp
            n = max(1, len(q) // 2)
            a, b = runtime.blend_curve_head(bundle, domain, s, q.iloc[:n], raw[:n], base[:n])
            np.testing.assert_allclose(a, rr[:n], rtol=0, atol=1e-06)
            np.testing.assert_allclose(b, pp[:n], rtol=0, atol=1e-06)
            try:
                runtime.blend_curve_head(bundle, domain, s, q.assign(true_soh=-999.0), raw, base)
            except ValueError:
                checks += 1
            else:
                raise AssertionError('Domain branch accepted query labels')
    actual['arm'] = CONTEXT['arm']
    assert len(actual) == 58214 and actual.battery_id.nunique() == 75
    assert np.isfinite(actual[['true_soh', 'pred_soh', 'raw_pred_soh']].to_numpy()).all()
    assert actual.pred_soh.between(0, 1.1).all()
    return (actual, checks)

def evaluate(arm, files):
    parts = [pd.read_csv(p, float_precision='round_trip').sort_values(['battery_id', 'cycle_index']).reset_index(drop=True) for p in files]
    identity = ['battery_id', 'cycle_index', 'true_soh', 'domain']
    for p in parts[1:]:
        pd.testing.assert_frame_equal(p[identity], parts[0][identity], check_exact=True)
    f = parts[0].copy()
    f['seed'] = 0
    for col in ['pred_soh', 'raw_pred_soh']:
        f[col] = sum((p[col].to_numpy() for p in parts)) / 3
    f.to_csv(OUT / arm / 'predictions_ensemble.csv', index=False)
    rows = []
    for (domain, bid), g in f.groupby(['domain', 'battery_id']):
        rows.append(dict(arm=arm, domain=domain, battery_id=bid, queries=len(g), **回归指标(g.true_soh, g.pred_soh), bias=float((g.pred_soh - g.true_soh).mean()), predicted_increases=int((np.diff(g.pred_soh) > 1e-12).sum())))
    battery = pd.DataFrame(rows)
    battery.to_csv(OUT / arm / 'battery_metrics.csv', index=False)
    domains = []
    for domain, g in battery.groupby('domain'):
        domains.append(dict(arm=arm, domain=domain, batteries=len(g), **{k: float(g[k].mean()) for k in ['mae', 'rmse', 'r2', 'bias']}))
    pd.DataFrame(domains).to_csv(OUT / arm / 'domain_metrics.csv', index=False)
    pooled = 回归指标(f.true_soh, f.pred_soh)
    row = dict(arm=arm, label=LABELS[arm], **{'pooled_' + k: v for k, v in pooled.items()}, **{'battery_macro_' + k: float(battery[k].mean()) for k in ['mae', 'rmse', 'r2']}, pct_error_le_1pp=float((abs(f.pred_soh - f.true_soh) <= 0.01).mean() * 100), predicted_increases=int(battery.predicted_increases.sum()))
    return (row, battery, domains)

def main():
    global UNIFORM_LR, UNIFORM_EPOCHS
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    cfg0 = torch.load(RELEASE / 'checkpoints/main/seed_20260903.pt', map_location='cpu', weights_only=True)['config']
    UNIFORM_LR = float(cfg0['目标学习率'])
    UNIFORM_EPOCHS = int(cfg0['目标训练轮数'])
    assert UNIFORM_LR == 0.0005 and UNIFORM_EPOCHS == 3
    sites = projection_sites()
    plan = register(sites)
    for r in sites:
        sys.modules[r['module']].np = NumpyView(r['module'])
    np.testing.assert_array_equal(MinimumProxy('test')([1, 3], [2, 2]), [1, 2])
    assert np.array_equal(np.minimum.accumulate([2, 1, 3]), [2, 1, 1])
    adaptation.仅训练预测头 = adapt_matched
    target = cohort(RELEASE)
    features = feature_tables(RELEASE)
    temperatures = json.loads((PROJECT / 'configs/nca_protocol_metadata.json').read_text(encoding='utf-8'))
    summaries = []
    batteries = {}
    domain_rows = []
    checks = []
    for arm in ARMS:
        CONTEXT['arm'] = arm
        folder = OUT / arm
        folder.mkdir(exist_ok=True)
        files = []
        for seed in SEEDS:
            CONTEXT['seed'] = seed
            bundle = torch.load(RELEASE / 'checkpoints/main' / f'seed_{seed}.pt', map_location='cpu', weights_only=True)
            actual, count = infer(bundle, target, features, temperatures, seed)
            gold = pd.read_csv(PROJECT / 'results/reference/predictions/main' / f'candidate_predictions_{seed}.csv', float_precision='round_trip')
            for k in ['battery_id', 'cycle_index', 'true_soh']:
                np.testing.assert_array_equal(actual[k], gold[k])
            if arm == 'full_method':
                for k in ['pred_soh', 'raw_pred_soh']:
                    np.testing.assert_allclose(actual[k], gold[k], rtol=0, atol=1e-06)
            file = folder / f'predictions_{seed}.csv'
            actual.to_csv(file, index=False)
            files.append(file)
            checks.append(dict(arm=arm, seed=seed, batteries=75, queries=58214, label_rejection_and_prefix_checks=count, canonical_bit_exact=arm == 'full_method' and all((np.array_equal(actual[k], gold[k]) for k in ['pred_soh', 'raw_pred_soh']))))
            print(json.dumps(checks[-1]), flush=True)
        row, b, ds = evaluate(arm, files)
        summaries.append(row)
        batteries[arm] = b
        domain_rows += ds
        if arm == 'full_method':
            stored = pd.read_csv(folder / 'predictions_ensemble.csv', float_precision='round_trip').sort_values(['battery_id', 'cycle_index'])
            expected = pd.read_csv(PROJECT / 'results/reference/predictions/ablations/full_method/predictions_ensemble.csv', float_precision='round_trip').sort_values(['battery_id', 'cycle_index'])
            np.testing.assert_allclose(stored.pred_soh.to_numpy(), expected.pred_soh.to_numpy(), rtol=0, atol=1e-06)
            torch.save(dict(rng=RNG_BANK, support_sha256=SUPPORT_BANK), OUT / 'matched_rng_states.pt')
    assert len(RNG_BANK) == 3 * 3 * 75
    result = pd.DataFrame(summaries)
    base = result[result.arm == 'full_method'].iloc[0]
    for metric in ['pooled_mae', 'pooled_rmse', 'pooled_r2', 'battery_macro_mae', 'battery_macro_rmse', 'battery_macro_r2']:
        result[metric + '_delta'] = result[metric] - base[metric]
    result['mae_relative_change_pct'] = (result.pooled_mae / base.pooled_mae - 1) * 100
    result.to_csv(OUT / 'ablation_summary.csv', index=False)
    pd.DataFrame(domain_rows).to_csv(OUT / 'ablation_domain_metrics.csv', index=False)
    pairs = []
    ref = batteries['full_method']
    for arm in ARMS[1:]:
        merged = batteries[arm].merge(ref, on=['domain', 'battery_id'], suffixes=('_ablated', '_full'), validate='one_to_one')
        for metric in ['mae', 'rmse', 'r2']:
            merged[metric + '_delta'] = merged[metric + '_ablated'] - merged[metric + '_full']
        merged.insert(0, 'intervention', arm)
        pairs.append(merged)
    pd.concat(pairs, ignore_index=True).to_csv(OUT / 'paired_battery_effects.csv', index=False)
    pd.DataFrame(ADAPTATION_LOG).to_csv(OUT / 'adaptation_budget_audit.csv', index=False)
    projection = [dict(arm=k[0], module=k[1], **v) for k, v in PROJECTION_LOG.items()]
    pd.DataFrame(projection).to_csv(OUT / 'projection_operator_audit.csv', index=False)
    assert all((v['disabled_calls'] == 0 for (arm, _), v in PROJECTION_LOG.items() if arm != 'no_monotonic_projection'))
    assert all((v['disabled_calls'] > 0 for (arm, _), v in PROJECTION_LOG.items() if arm == 'no_monotonic_projection'))
    audit = pd.DataFrame(ADAPTATION_LOG)
    for arm in ARMS:
        observed = audit[(audit.arm == arm) & (audit.shots == 20)]
        assert len(observed) == 225
        if arm == 'uniform_support_budget':
            assert observed.effective_lr.eq(UNIFORM_LR).all() and observed.effective_epochs.eq(UNIFORM_EPOCHS).all()
        else:
            assert observed.effective_lr.eq(observed.requested_lr).all() and observed.effective_epochs.eq(observed.requested_epochs).all()
    assert verify(RELEASE)['passed']
    write(OUT / 'integrity.json', dict(passed=True, arms=list(ARMS), component_replays=checks, full_method_bit_exact=all((v['canonical_bit_exact'] for v in checks if v['arm'] == 'full_method')), full_ensemble_matches_frozen_primary_result=True, matched_rng_entries=len(RNG_BANK), encoder_frozen_during_all_support_adaptation=True, projection_modules=len(sites), projection_sites=sum((len(r['lines']) for r in sites)), clipping_preserved=True, no_source_or_domain_refit=True, model_and_source_hashes_unchanged=True, policy_search_executed=False, main_model_replaced=False))
    print(result[['label', 'pooled_mae', 'pooled_rmse', 'pooled_r2', 'battery_macro_r2', 'mae_relative_change_pct']].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
