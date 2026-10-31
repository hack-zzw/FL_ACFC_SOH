"""Verify, replay and score the user-selected, frozen 20-shot release."""
from __future__ import annotations
from utils.paths import ROOT as PACKAGE_ROOT
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
PROJECT = PACKAGE_ROOT
DEFAULT_RELEASE = PROJECT
SEEDS = (20260903, 20260904, 20260905)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(release=DEFAULT_RELEASE):
    manifest = json.loads((PROJECT / 'configs/files.json').read_text(encoding='utf-8'))
    for item in manifest:
        p = (PROJECT / item['path']).resolve()
        if not p.is_relative_to(PROJECT) or digest(p) != item['sha256']:
            raise ValueError(f"Release checksum mismatch: {item['path']}")
    return dict(passed=True, files=len(manifest), release=release.name)

def cohort(release):
    from dataloader.loader import 读取数据
    f = 读取数据(PROJECT / 'data/features', 划分种子=20260903)
    f['domain'] = f.material + '|' + f.source
    actual = f.groupby(['battery_id', 'domain', 'client_id', 'partition'], as_index=False).agg(records=('soh', 'size'))
    expected = pd.read_csv(PROJECT / 'configs/data/main/battery_manifest.csv')
    pd.testing.assert_frame_equal(actual.sort_values('battery_id').reset_index(drop=True), expected.sort_values('battery_id').reset_index(drop=True), check_dtype=False)
    return f[f.partition == 'target_test'].copy()

def feature_tables(release=None):
    a = PROJECT / 'data/windows/target'
    tables = {d: pd.read_pickle(a / f'{n}.pkl') for d, n in {'NCA|TJU': 'NCA_window', 'NCM|XJTU': 'XJTU_window', 'LFP|HUST': 'HUST_window', 'LFP|MIT': 'MIT_quality'}.items()}
    qv = pd.concat([pd.read_pickle(a / 'NCA_QV.pkl'), pd.read_pickle(a / 'NCA_halfC_QV.pkl')], ignore_index=True)
    low = pd.read_pickle(a / 'XJTU_low_voltage.pkl')
    tables['NCA|TJU'] = tables['NCA|TJU'].merge(qv, on=['battery_id', 'cycle_index'], how='left', validate='one_to_one')
    tables['NCM|XJTU'] = tables['NCM|XJTU'].merge(low, on=['battery_id', 'cycle_index'], validate='one_to_one')
    return tables

def replay(release, output, seeds=SEEDS, checks=True, tolerance=1e-06):
    from configs.schema import 实验配置
    from models.adaptation import _评估
    from utils.state import restore_rng, restore_scaler
    from models.domain.reference_head import apply_reference
    from models.domain import inference as runtime
    if not torch.cuda.is_available():
        raise RuntimeError('Exact archived replay requires the pytorch CUDA environment; CPU parity is not established.')
    verify(release)
    output = output.resolve()
    if not output.is_relative_to((PROJECT / 'results/generated').resolve()):
        raise ValueError('Write generated replay files under the package outputs directory.')
    output.mkdir(parents=True, exist_ok=True)
    target = cohort(release)
    features = feature_tables(release)
    temperatures = json.loads((PROJECT / 'configs/nca_protocol_metadata.json').read_text(encoding='utf-8'))
    reports = []
    for seed in seeds:
        bundle = torch.load(PROJECT / 'checkpoints/main' / f'seed_{seed}.pt', map_location='cpu', weights_only=True)
        cfg = 实验配置(**bundle['config'])
        scaler = restore_scaler(bundle['scaler'])
        restore_rng(bundle)
        for shots in cfg.shots:
            _, _, rows = _评估(bundle['method'], bundle['global_state'], bundle['domain_states'], target, scaler, copy.deepcopy(cfg), torch.device('cuda'), seed, shots)
            if shots == 20:
                actual = pd.DataFrame(rows)
                actual['domain'] = actual.material + '|' + actual.source
        for domain, reference in bundle['reference_heads'].items():
            idx = actual.index[actual.domain == domain]
            f = apply_reference(actual.loc[idx], target[target.domain == domain], reference)
            actual.loc[idx, ['pred_soh', 'raw_pred_soh']] = f[['pred_soh', 'raw_pred_soh']]
        count = 0
        if 'ordered_quality_residuals' in bundle:
            held = target[target.domain == 'LFP|MIT'].merge(features['LFP|MIT'], on=['battery_id', 'cycle_index'], validate='one_to_one')
            for bid, g in held.groupby('battery_id', sort=True):
                g = g.sort_values('cycle_index')
                s = g.iloc[:20]
                q = g.iloc[20:].drop(columns=['soh', 'capacity'])
                idx = actual.index[actual.battery_id == bid]
                raw = actual.loc[idx].raw_pred_soh.to_numpy().copy()
                base = actual.loc[idx].pred_soh.to_numpy().copy()
                p = runtime.correct_mit(bundle, s, q, raw, base)
                actual.loc[idx, 'pred_soh'] = p
                if checks:
                    n = max(1, len(q) // 2)
                    np.testing.assert_allclose(runtime.correct_mit(bundle, s, q.iloc[:n], raw[:n], base[:n]), p[:n], rtol=0, atol=1e-12)
                    try:
                        runtime.correct_mit(bundle, s, q.assign(true_soh=-999.0), raw, base)
                    except ValueError:
                        count += 1
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
                if checks:
                    n = max(1, len(q) // 2)
                    a, b = runtime.blend_curve_head(bundle, domain, s, q.iloc[:n], raw[:n], base[:n])
                    np.testing.assert_allclose(a, rr[:n], rtol=0, atol=1e-06)
                    np.testing.assert_allclose(b, pp[:n], rtol=0, atol=1e-06)
                    try:
                        runtime.blend_curve_head(bundle, domain, s, q.assign(true_soh=-999.0), raw, base)
                    except ValueError:
                        count += 1
                    else:
                        raise AssertionError('Curve adapter accepted query labels')
        actual['arm'] = 'candidate'
        expected = pd.read_csv(PROJECT / 'results/reference/predictions/main' / f'candidate_predictions_{seed}.csv', float_precision='round_trip')
        assert len(actual) == 58214 and actual.battery_id.nunique() == 75
        for col in ['battery_id', 'cycle_index', 'true_soh']:
            np.testing.assert_array_equal(actual[col], expected[col], err_msg=f'{seed}: {col}')
        errors = {col: float(np.max(abs(actual[col].to_numpy() - expected[col].to_numpy()))) for col in ['pred_soh', 'raw_pred_soh']}
        for col in errors:
            np.testing.assert_allclose(actual[col], expected[col], rtol=0, atol=tolerance, err_msg=f'{seed}: {col}')
        actual.to_csv(output / f'actual_replay_{seed}.csv', index=False)
        reports.append(dict(seed=seed, batteries=75, queries=len(actual), bit_exact=max(errors.values()) == 0.0, maximum_prediction_difference=max(errors.values()), prediction_atol=tolerance, label_rejection_and_prefix_checks=count))
        print(json.dumps(reports[-1]), flush=True)
    result = dict(passed=True, replays=reports, support_labels=20)
    if tuple(seeds) == SEEDS:
        result['metrics'] = score(release, output).to_dict(orient='records')
    (output / 'integrity.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result

def score(release, output):
    from utils.statistics import average_components, summarize, metric_roundoff_bounds
    frames = [pd.read_csv(output / f'actual_replay_{seed}.csv', float_precision='round_trip') for seed in SEEDS]
    prediction = average_components(frames)
    battery, domain, pooled = summarize(prediction)
    expected = pd.read_csv(PROJECT / 'results/reference/metrics/paper/table1_primary_domain.csv', float_precision='round_trip')
    reference = average_components([pd.read_csv(PROJECT / 'results/reference/predictions/main' / f'candidate_predictions_{seed}.csv', float_precision='round_trip') for seed in SEEDS])
    _, reference_domain, _ = summarize(reference)
    for row in expected.itertuples():
        actual = domain[domain.domain.eq(row.domain)].iloc[0]
        refrow = reference_domain[reference_domain.domain.eq(row.domain)].iloc[0]
        group = prediction[prediction.domain.eq(row.domain)]
        ref = reference[reference.domain.eq(row.domain)]
        bounds = [metric_roundoff_bounds(g, ref[ref.battery_id.eq(bid)]) for bid, g in group.groupby('battery_id')]
        for name in ['mae', 'rmse', 'r2', 'mape']:
            assert abs(refrow[name] - getattr(row, name)) < 1e-12, (row.domain, name)
            assert abs(actual[name] - getattr(row, name)) <= float(np.mean([b[name] for b in bounds])) + 1e-12, (row.domain, name)
    prediction.to_csv(output / 'ensemble_predictions.csv.gz', index=False)
    battery.to_csv(output / 'battery_metrics.csv', index=False)
    domain.to_csv(output / 'five_domain_metrics.csv', index=False)
    (output / 'pooled_metrics.json').write_text(json.dumps(pooled, indent=2), encoding='utf-8')
    return domain

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'replay', 'score'])
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/generated/selected_release_replay')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Replay one component; omit for the full three-component release.')
    parser.add_argument('--shots', type=int, choices=[20], default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'verify':
        result = verify(args.release)
    elif args.command == 'score':
        result = score(args.release, args.output).to_dict(orient='records')
    else:
        result = replay(args.release, args.output, (args.seed,) if args.seed else SEEDS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
