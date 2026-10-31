"""Compute paper statistics exclusively from the current model-replay outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils.paths import ROOT
from utils.statistics import (SEEDS, KEY, read, summarize, average_components,
                              primary_intervals, assert_numeric, metric,
                              metric_roundoff_bounds)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def scored_case(frame, method, variant):
    battery, domain, pooled = summarize(frame)
    battery = battery.rename(columns={'queries': 'query_records'}).assign(method=method, variant=variant)
    domain = domain.rename(columns={k: 'battery_macro_' + k for k in ['mae', 'rmse', 'r2', 'mape']}).assign(method=method, variant=variant)
    return battery, domain, pooled


def native_predictions():
    """Frozen native ensembles used only by the prediction equality check."""
    folder = ROOT / 'results/reference/predictions/native'
    domains = read(ROOT / 'configs/data/main/battery_manifest.csv').set_index('battery_id').domain
    for config in ['centralized', 'fedavg', 'material_balanced', 'fedprox_mu_1e-4', 'fedprox_mu_1e-2', 'fedprox_mu_1e-1', 'clustered']:
        if config == 'clustered':
            parts = [read(folder / f'clustered_{seed}.csv.gz') for seed in SEEDS]
        else:
            frame = read(folder / f'{config}.csv.gz')
            parts = [frame[frame.seed.eq(seed)].copy() for seed in SEEDS]
        for frame in parts:
            frame['domain'] = frame.battery_id.map(domains)
        yield config, average_components(parts)


def check_external(actual, expected_predictions, expected_metrics, dataset):
    """Bound external CPU roundoff by its measured effect on each metric."""
    pred = 'prediction' if dataset == 'NASA' else 'raw_prediction'
    values = []
    maximum = 0.
    allowed = 0.
    for shots, group in actual.groupby('shots'):
        expected = expected_predictions[expected_predictions.shots.eq(shots)]
        a = group.rename(columns={'soh': 'true_soh', pred: 'pred_soh'})
        b = expected.rename(columns={'soh': 'true_soh', pred: 'pred_soh'})
        if dataset == 'NASA':
            result = metric(group.soh, group[pred])
            bounds = metric_roundoff_bounds(a, b)
        else:
            result = {k: float(np.mean([metric(h.soh, h[pred])[k] for _, h in group.groupby('battery_id')])) for k in ['mae', 'rmse', 'r2', 'mape']}
            bs = [metric_roundoff_bounds(h, b[b.battery_id.eq(bid)]) for bid, h in a.groupby('battery_id')]
            bounds = {k: float(np.mean([v[k] for v in bs])) for k in result}
        reference = expected_metrics[expected_metrics.shots.eq(shots)].iloc[0]
        for key, value in result.items():
            diff = abs(value - float(reference[key]))
            bound = bounds[key] + 1e-12
            if diff > bound:
                raise AssertionError((dataset, shots, key, diff, bound))
            maximum = max(maximum, diff)
            allowed = max(allowed, bound)
        values.append(dict(shots=shots, **result))
    return pd.DataFrame(values), dict(experiment=dataset, max_difference=maximum, maximum_roundoff_bound=allowed)


def main(output, figures=True):
    out = Path(output)
    paper = out / 'metrics/paper'
    paper.mkdir(parents=True, exist_ok=True)
    ref = ROOT / 'results/reference/metrics/paper'
    figure_inputs = out / 'figure_inputs'
    figure_inputs.mkdir(exist_ok=True)
    current_inputs = []

    def current(relative):
        path = out / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing current-run prediction: {path}. Run python main.py.')
        current_inputs.append(path)
        return read(path)

    checks = []
    primary = average_components([current(f'main/actual_replay_{s}.csv') for s in SEEDS])
    battery, domain, pooled = summarize(primary)
    checks.append(dict(experiment='main', max_difference=assert_numeric(domain, read(ref / 'table1_primary_domain.csv'), ['domain'], ['mae', 'rmse', 'r2', 'mape'], 1e-12)))
    ci = primary_intervals(battery)
    checks.append(dict(experiment='main_battery_bootstrap', max_difference=assert_numeric(ci, read(ref / 'conditional_battery_bootstrap_CI.csv'), ['domain', 'metric'], ['estimate', 'low95', 'high95'], 1e-12)))
    save(primary, out / 'predictions/main_ensemble.csv.gz')
    save(battery, paper / 'primary_battery_metrics.csv')
    save(domain, paper / 'table1_primary_domain.csv')
    save(ci, paper / 'conditional_battery_bootstrap_CI.csv')
    (paper / 'pooled_metrics.json').write_text(json.dumps(pooled, indent=2), encoding='utf-8')

    table = [dict(configuration='primary', pooled_mae=pooled['mae'], pooled_rmse=pooled['rmse'], pooled_r2=pooled['r2'], battery_macro_r2=pooled['battery_macro_r2'])]
    display = {'centralized': 'Centralized (17D)', 'fedavg': 'FedAvg', 'fedprox_mu_1e-4': 'FedProx',
               'material_balanced': 'Material heads', 'clustered': 'Material × clusters'}
    plotted_native = [primary.assign(method='Paper method')]
    for config in ['centralized', 'fedavg', 'material_balanced', 'fedprox_mu_1e-4', 'fedprox_mu_1e-2', 'fedprox_mu_1e-1', 'clustered']:
        frame = current(f'native/{config}__ensemble.csv.gz')
        _, _, scores = summarize(frame)
        table.append(dict(configuration=config, pooled_mae=scores['mae'], pooled_rmse=scores['rmse'], pooled_r2=scores['r2'], battery_macro_r2=scores['battery_macro_r2']))
        if config in display:
            plotted_native.append(frame.assign(method=display[config]))
    table = pd.DataFrame(table)
    checks.append(dict(experiment='Table1_all_eight_configurations', max_difference=assert_numeric(table, read(ROOT / 'results/source_data/Table1.csv'), ['configuration'], ['pooled_mae', 'pooled_rmse', 'pooled_r2', 'battery_macro_r2'], 1e-12)))
    save(table, out / 'source_data/Table1.csv')
    native = pd.concat(plotted_native, ignore_index=True)[KEY + ['true_soh', 'pred_soh', 'domain', 'method']]
    save(primary, figure_inputs / 'primary_predictions.csv.gz')
    save(native, figure_inputs / 'native_predictions.csv.gz')

    thresholds = []
    for name, group in primary.groupby('domain', sort=True):
        for limit_pp in [.5, 1., 2., 3.]:
            thresholds.append(dict(domain=name, threshold_pp=limit_pp,
                                   percent=float(np.mean(abs(group.pred_soh-group.true_soh) <= limit_pp/100)*100)))
    save(pd.DataFrame(thresholds), paper / 'absolute_error_thresholds.csv')

    from utils.bootstrap import material_pairs, contrast, factorial
    material = out / 'metrics/material_readout'
    material.mkdir(parents=True, exist_ok=True)
    batteries, domains, overalls = [], [], []
    for method in ['M1_shared_mapping', 'Mmid_material_mapping', 'M2_domain_mapping']:
        for variant in ['projected', 'clip_only']:
            frame = current(f'readouts_replay/{method}__{variant}__ensemble.csv.gz')
            bb, dd, oo = scored_case(frame, method, variant)
            batteries.append(bb)
            domains.append(dd)
            overalls.append(dict(method=method, variant=variant, pooled_mae=oo['mae'], pooled_r2=oo['r2']))
    mb, md = pd.concat(batteries, ignore_index=True), pd.concat(domains, ignore_index=True)
    save(mb, material / 'battery_metrics.csv')
    save(md, material / 'domain_metrics.csv')
    save(pd.DataFrame(overalls), material / 'overall_metrics.csv')
    checks.append(dict(experiment='mapping_granularity', max_difference=assert_numeric(md, read(ROOT / 'results/reference/metrics/material_readout/domain_metrics.csv'), ['method', 'variant', 'domain'], ['battery_macro_mae', 'battery_macro_r2'])))
    material_pairs(mb, material)
    checks.append(dict(experiment='mapping_bootstrap', max_difference=assert_numeric(read(material / 'paired_bootstrap_intervals.csv'), read(ROOT / 'results/reference/metrics/material_readout/paired_bootstrap_intervals.csv'), ['variant', 'before', 'after', 'metric'], ['difference', 'pointwise_conditional_95_lower', 'pointwise_conditional_95_upper'])))

    control_batteries, effects = [], []
    comparisons = {'01_capacity': [('shared', 'material'), ('material', 'domain'), ('shared', 'domain')],
                   '02_representation': [('raw', 'learned'), ('random', 'learned')],
                   '03_semantic_groups': [(f'pseudo_{i}', 'true_domain') for i in range(1, 6)]}
    for stage in ['01_capacity', '02_representation', '03_semantic_groups', '04_federated']:
        rows = []
        paths = sorted((out / f'controls/{stage}/predictions').glob('*__ensemble.csv.gz'))
        if not paths:
            raise FileNotFoundError(f'Missing control predictions: {stage}')
        for path in paths:
            method, variant = path.name.removesuffix('__ensemble.csv.gz').split('__')
            bb, _, _ = scored_case(current(path.relative_to(out)), method, variant)
            rows.append(bb)
        bb = pd.concat(rows, ignore_index=True)
        control_batteries.append(bb.assign(stage=stage))
        checks.append(dict(experiment=stage, max_difference=assert_numeric(bb, read(ROOT / f'results/reference/metrics/controls/{stage}/battery_metrics.csv'), ['method', 'variant', 'battery_id'], ['mae', 'rmse', 'r2', 'mape'])))
        for variant in ['projected', 'clip_only']:
            for before, after in comparisons.get(stage, []):
                effects.extend(dict(stage=stage, **value) for value in contrast(bb, before, after, variant))
        if stage == '04_federated':
            factor = factorial(bb)
    controls = out / 'metrics/controls'
    controls.mkdir(parents=True, exist_ok=True)
    cb = pd.concat(control_batteries, ignore_index=True)
    effects = pd.DataFrame(effects)
    save(cb, controls / 'battery_metrics.csv')
    save(effects, controls / 'planned_paired_effects.csv')
    save(factor, controls / 'federated_factorial_paired_effects.csv')
    rr = ROOT / 'results/reference/metrics/controls'
    checks.append(dict(experiment='controlled_paired_bootstrap', max_difference=assert_numeric(effects, read(rr / 'planned_paired_effects.csv'), ['stage', 'reference', 'candidate', 'variant', 'metric'], ['paired_difference', 'conditional95_low', 'conditional95_high'])))
    checks.append(dict(experiment='federated_factorial_bootstrap', max_difference=assert_numeric(factor, read(rr / 'federated_factorial_paired_effects.csv'), ['variant', 'contrast'], ['battery_macro_mae_difference', 'conditional95_low', 'conditional95_high'])))

    interventions, ablation_frames = [], []
    for arm in ['full_method', 'uniform_support_budget', 'no_monotonic_projection']:
        frame = current(f'ablations/{arm}/predictions_ensemble.csv')
        bb, _, oo = summarize(frame)
        interventions.append(dict(arm=arm, pooled_mae=oo['mae'], pooled_rmse=oo['rmse'], pooled_r2=oo['r2'], battery_macro_r2=oo['battery_macro_r2'], predicted_increases=int(bb.predicted_increases.sum())))
        ablation_frames.append(frame.assign(experiment=arm))
    inter = pd.DataFrame(interventions)
    inter['mae_relative_change_pct'] = (inter.pooled_mae/inter[inter.arm.eq('full_method')].pooled_mae.iloc[0]-1)*100
    checks.append(dict(experiment='pipeline_interventions', max_difference=assert_numeric(inter, read(ROOT / 'results/reference/metrics/ablations/ablation_summary.csv'), ['arm'], ['pooled_mae', 'pooled_rmse', 'pooled_r2', 'mae_relative_change_pct', 'predicted_increases'])))
    save(inter, out / 'metrics/Results_2_4.csv')
    save(pd.concat(ablation_frames, ignore_index=True), out / 'source_data/Results_2_4.csv.gz')

    external_frames = []
    for dataset, filename, table_name in [('NASA', 'NASA_fixed_query', 'table5_NASA.csv'), ('Oxford', 'Oxford', 'table6_Oxford.csv')]:
        frame = current(f'external/{dataset}.csv.gz')
        expected_predictions = read(ROOT / f'results/reference/predictions/figure_inputs/{filename}.csv.gz')
        scores, check = check_external(frame, expected_predictions, read(ref / table_name), dataset)
        checks.append(check)
        save(scores, paper / table_name)
        save(frame, figure_inputs / f'{filename}.csv.gz')
        external_frames.append(frame.assign(dataset=dataset))
    save(pd.concat(external_frames, ignore_index=True), out / 'source_data/Fig4.csv.gz')

    # Fixed observed supports and illustrative battery IDs are study inputs;
    # every displayed prediction and error is generated in this run.
    fixed = ROOT / 'data/figure_selection'
    support = read(fixed / 'support_observations.csv')
    representatives = read(fixed / 'representative_batteries.csv')
    save(support, figure_inputs / 'support_observations.csv')
    save(representatives, figure_inputs / 'representative_batteries.csv')
    save(pd.concat([primary.assign(record_type='query_prediction', pipeline='Paper method'),
                    native.assign(record_type='pipeline_query'), battery.assign(record_type='battery_metrics'),
                    ci.assign(record_type='domain_interval'), support.assign(record_type='support_observation'),
                    representatives.assign(record_type='representative_battery')], ignore_index=True), out / 'source_data/Fig2.csv.gz')
    save(pd.concat([cb.assign(record_type='control_battery_metrics'), mb.assign(record_type='mapping_battery_metrics'),
                    md.assign(record_type='mapping_domain_means'), effects.assign(record_type='paired_interval'),
                    factor.assign(record_type='factorial_interval'), read(material / 'paired_bootstrap_intervals.csv').assign(record_type='mapping_interval'),
                    read(material / 'paired_battery_differences.csv').assign(record_type='mapping_paired_battery_difference')], ignore_index=True), out / 'source_data/Fig3.csv.gz')

    save(pd.DataFrame(checks), out / 'validation/numerical_checks.csv')
    provenance = [dict(path=str(p.relative_to(out)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in current_inputs]
    report = dict(passed=True, checks=len(checks), maximum_metric_difference=max(c['max_difference'] for c in checks),
                  main_pooled=pooled, model_inference_performed=True, reference_predictions_used_for_statistics=False,
                  statistics_input='current-run model predictions', current_prediction_files=provenance)
    (out / 'validation/result_recomputation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if figures:
        from utils import figures as plotting
        plotting.main()
    print(json.dumps({k: v for k, v in report.items() if k != 'current_prediction_files'}, indent=2), flush=True)
    return report
