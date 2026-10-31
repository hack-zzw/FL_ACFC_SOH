"""Saved-model prediction paths for every numerical experiment in the paper."""
import copy, dataclasses, importlib.util, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.paths import ROOT
from utils.statistics import *

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def compare(actual, expected, tolerance=1e-06):
    a = actual.sort_values(KEY).reset_index(drop=True)
    b = expected.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[KEY], b[KEY], check_dtype=False)
    diff = float(np.max(abs(a.pred_soh.to_numpy() - b.pred_soh.to_numpy())))
    assert diff <= tolerance, (diff, tolerance)
    return diff

def write_report(out, suite, rows):
    d = out / 'validation'
    d.mkdir(exist_ok=True)
    (d / f'{suite}_replay.json').write_text(json.dumps(dict(passed=True, checks=rows), indent=2), encoding='utf-8')

def native(out):
    from configs.schema import 实验配置
    from dataloader.loader import 读取数据, 按电池划分训练验证
    from dataloader.preprocessing import 拟合材料标准化器
    from utils.runtime import 设置随机种子, 选择设备
    from models.comparisons import _评估策略
    from utils.results import native_predictions
    frame = 读取数据(ROOT / 'data/features', 划分种子=20260903)
    source, _ = 按电池划分训练验证(frame)
    target = frame[frame.partition.eq('target_test')].copy()
    scaler = 拟合材料标准化器(source)
    expected = dict(native_predictions())
    rows = []
    folder = out / 'native'
    folder.mkdir(exist_ok=True)
    for configuration in ['centralized', 'fedavg', 'material_balanced', 'fedprox_mu_1e-4', 'fedprox_mu_1e-2', 'fedprox_mu_1e-1', 'clustered']:
        parts = []
        for seed in SEEDS:
            strategy = 'fedprox' if configuration.startswith('fedprox') else configuration
            sub = Path(configuration) / f'fedprox_seed_{seed}.pt' if strategy == 'fedprox' else Path(f'{strategy}_seed_{seed}.pt')
            saved = torch.load(ROOT / 'checkpoints/native' / sub, map_location='cpu', weights_only=True)
            cfg = 实验配置(**{k: v for k, v in saved['config'].items() if k in {f.name for f in dataclasses.fields(实验配置)}})
            t = target.copy()
            if strategy == 'clustered':
                router = json.loads((ROOT / f'configs/native/cluster_router_{seed}.json').read_text(encoding='utf-8'))
                prefix = t.sort_values(KEY).groupby('battery_id').head(20)
                means = prefix.groupby('battery_id')[router['features']].mean().reindex(columns=router['features'])
                routes = means.index.to_series().map(router['battery_routes']) if 'battery_routes' in router else None
                if routes is None:
                    from utils.cluster_router import causal_routes
                    routes = causal_routes(t, router)
                t['原始source'] = t.source.astype(str)
                t['source'] = t.battery_id.map(routes)
            设置随机种子(seed + 100000 + 20)
            _, _, pred = _评估策略(strategy, saved['global_state'], saved['material_states'], t, scaler, cfg, 选择设备(cfg.设备), seed, 20, 域策略对所有策略生效=cfg.额外参数.get('域策略对所有策略生效', True))
            f = pd.DataFrame(pred)
            f['domain'] = f.battery_id.map(target.groupby('battery_id').apply(lambda g: str(g.material.iloc[0]) + '|' + str(g.source.iloc[0]), include_groups=False))
            f.to_csv(folder / f'{configuration}_{seed}.csv.gz', index=False)
            parts.append(f)
        f = average_components(parts)
        diff = compare(f, expected[configuration])
        f.to_csv(folder / f'{configuration}__ensemble.csv.gz', index=False)
        rows.append(dict(configuration=configuration, maximum_prediction_difference=diff, queries=len(f)))
        print(json.dumps(rows[-1]), flush=True)
    write_report(out, 'native', rows)

def readouts(out):
    c = module(ROOT / 'models/readouts/common.py', 'ridge_common')
    prep = ROOT / 'data/prepared'
    with (prep / 'target_tasks.pkl').open('rb') as f:
        ts = pickle.load(f)
    numeric = np.load(prep / 'target_numeric.npy', mmap_mode='r')
    truth = read(prep / 'target_truth.csv').sort_values(KEY).reset_index(drop=True)
    folder = out / 'readouts_replay'
    folder.mkdir(exist_ok=True)
    rows = []
    for arm in ['M1_shared_mapping', 'Mmid_material_mapping', 'M2_domain_mapping']:
        parts = {v: [] for v in ['projected', 'clip_only']}
        for seed in SEEDS:
            model = c.load_model(ROOT / f'checkpoints/readouts/{arm}/model_{seed}.npz')
            emb = np.load(prep / f'target_embedding_{seed}.npy', mmap_mode='r')
            x = np.column_stack([numeric, emb])
            local = {v: [] for v in parts}
            for t in ts:
                raw, p = c.predict_task(model, x[t['start']:t['end']], t, 'projected')
                for variant in local:
                    pred = p if variant == 'projected' else c.post(raw, t['support_last'], variant)
                    local[variant].append(pd.DataFrame(dict(battery_id=t['battery_id'], cycle_index=t['cycle_index'][20:], domain=t['domain'], pred_soh=pred, raw_pred_soh=raw)))
            for variant in local:
                f = pd.concat(local[variant], ignore_index=True).sort_values(KEY).reset_index(drop=True)
                pd.testing.assert_frame_equal(f[KEY], truth[KEY], check_dtype=False)
                f['true_soh'] = truth.true_soh.to_numpy()
                parts[variant].append(f)
                f.to_csv(folder / f'{arm}__{variant}__{seed}.csv.gz', index=False)
        for variant in parts:
            f = average_components(parts[variant])
            ref_folder = 'material_readout' if arm == 'Mmid_material_mapping' else 'readouts'
            diff = compare(f, read(ROOT / f'results/reference/predictions/{ref_folder}/{arm}__{variant}.csv.gz'), 1e-12)
            f.to_csv(folder / f'{arm}__{variant}__ensemble.csv.gz', index=False)
            rows.append(dict(arm=arm, variant=variant, maximum_prediction_difference=diff))
    write_report(out, 'readouts', rows)
    print('Ridge readout replay passed', flush=True)

def controls(out):
    c = module(ROOT / 'models/controls/common.py', 'common')
    readout = module(ROOT / 'models/controls/run_readouts.py', 'control_readouts')
    inventory = read(ROOT / 'configs/control_models.csv')
    for r in inventory.itertuples():
        source = ROOT / r.path
        rel = Path(r.stage) / r.method / f'seed_{r.seed}'
        rel = rel / 'model.pt' if r.stage == '04_federated' else rel / 'all' / f'epoch_{int(r.selected_epoch)}.pt'
        dest = c.ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        (dest.parent / 'completion.json').write_text('{"passed":true}', encoding='utf-8')
    for stage in ['01_capacity', '02_representation', '03_semantic_groups']:
        done = c.ROOT / stage / 'evaluation_completion.json'
        if done.exists():
            done.unlink()
        selections = json.loads((ROOT / f'configs/controls/{stage}/source_selections.json').read_text(encoding='utf-8'))
        c.dump(c.ROOT / stage / 'source_selections.json', selections)
        c.evaluate_heads(stage, readout.configs_for(stage), selections)
    done = c.ROOT / '04_federated/evaluation_completion.json'
    if done.exists():
        done.unlink()
    fl = module(ROOT / 'models/controls/run_federated.py', 'control_federated')
    fl.evaluate(c.tasks('source'))
    rows = []
    for stage in ['01_capacity', '02_representation', '03_semantic_groups', '04_federated']:
        for p in sorted((c.ROOT / stage / 'predictions').glob('*__ensemble.csv.gz')):
            diff = compare(read(p), read(ROOT / f'results/reference/predictions/controls/{stage}' / p.name))
            rows.append(dict(stage=stage, prediction_file=p.name, maximum_prediction_difference=diff))
    write_report(out, 'controls', rows)

def external(out):
    """Predict external fixed queries; reference predictions are comparison only."""
    from types import SimpleNamespace
    from models.external.source import read_config
    from models.external.nasa_pipeline import clients_for
    from models.external.nasa_training import restore
    from models.external.nasa_prediction import option_predict
    from models.external.adaptation import input_features
    from models.external.predict import predict_dataset_client
    from dataloader.external import 读取外部数据清单
    cfg = read_config(ROOT / 'configs/best_no_pinn_experiment_config.json')
    clients = clients_for(SimpleNamespace(manifests=ROOT / 'data/external/manifests'))
    protocol = json.loads((ROOT / 'configs/external/NASA_external_protocol.json').read_text(encoding='utf-8'))
    plan = read(ROOT / 'data/external/NASA_queries.csv')
    reference = read(ROOT / 'results/reference/external/NASA/personalization_predictions.csv')
    folder = out / 'external'
    folder.mkdir(exist_ok=True)
    rows, nasa = ([], [])
    identity = ['client', 'battery_id', 'record_index']
    for shot in [20, 40, 60]:
        option = protocol['support_residual_policies'][str(shot)]
        parts = []
        for seed in SEEDS:
            actual = []
            for client, queries in plan[plan.shots.eq(shot)].groupby('client', sort=True):
                local = torch.load(ROOT / f'checkpoints/external/NASA/personalized/{shot}_{seed}_{client}_{option}.pt', map_location='cpu', weights_only=True)
                model = restore(local['model'], cfg, 'cpu', seed)
                full = clients[client].sort_values(KEY).copy()
                full['record_index'] = full.groupby('battery_id').cumcount()
                q = queries[['battery_id', 'record_index']].merge(full, on=['battery_id', 'record_index'], validate='one_to_one')
                pred, _ = option_predict(model, pd.DataFrame(local['support']), input_features(q), local['head'], option, 'cpu')
                h = q[['battery_id', 'cycle_index', 'record_index', 'soh']].copy()
                h['prediction'] = pred
                h['client'], h['shots'], h['seed'] = (client, shot, seed)
                h['option'], h['query_mode'] = ('selected', 'matched_long_60to79')
                expected = reference[reference.option.eq('selected') & reference.shots.eq(shot) & reference.seed.eq(seed) & reference.query_mode.eq('matched_long_60to79') & reference.client.eq(client)]
                aligned = h.merge(expected[['battery_id', 'record_index', 'prediction']], on=['battery_id', 'record_index'], suffixes=('_actual', '_reference'), validate='one_to_one')
                assert len(aligned) == len(h) == len(expected)
                diff = float(np.max(abs(aligned.prediction_actual - aligned.prediction_reference)))
                assert diff < 1e-06
                rows.append(dict(dataset='NASA', shots=shot, seed=seed, client=client, maximum_prediction_difference=diff))
                actual.append(h)
            parts.append(pd.concat(actual, ignore_index=True).sort_values(identity).reset_index(drop=True))
        for frame in parts[1:]:
            pd.testing.assert_frame_equal(frame[identity + ['soh']], parts[0][identity + ['soh']], check_dtype=False)
        ensemble = parts[0].copy()
        ensemble['prediction'] = np.mean([f.prediction.to_numpy() for f in parts], axis=0)
        ensemble['seed'], ensemble['ensemble_components'] = (0, 3)
        nasa.append(ensemble)
    pd.concat(nasa, ignore_index=True).to_csv(folder / 'NASA.csv.gz', index=False)
    full = 读取外部数据清单(ROOT / 'data/external/manifests/Oxford_10.json').sort_values(KEY)
    full['record_index'] = full.groupby('battery_id').cumcount()
    plan = read(ROOT / 'data/external/Oxford_queries.csv')
    reference = read(ROOT / 'results/reference/external/Oxford/evaluation_predictions.csv')
    oxford = []
    for shot in [0, 5, 10, 20]:
        actual_shot = 20 if shot == 0 else shot
        arm = 'zero_shot' if shot == 0 else 'dataset_specific_acfc'
        queries = plan[plan.shots.eq(shot)]
        q = queries[['battery_id', 'record_index']].merge(full, on=['battery_id', 'record_index'], validate='one_to_one')
        parts = []
        for seed in SEEDS:
            path = ROOT / f'checkpoints/external/Oxford/{arm}_{actual_shot}_{seed}.pt'
            pred = predict_dataset_client(path, input_features(q), torch.device('cpu'))
            h = q[['battery_id', 'cycle_index', 'record_index', 'soh']].copy()
            h['raw_prediction'] = pred
            h['shots'], h['seed'], h['arm'] = (shot, seed, arm)
            h['client'] = 'Oxford_10'
            expected = reference[reference.arm.eq(arm) & reference.shots.eq(actual_shot) & reference.seed.eq(seed) & reference.record_index.ge(20)]
            aligned = h.merge(expected[['battery_id', 'record_index', 'raw_prediction']], on=['battery_id', 'record_index'], suffixes=('_actual', '_reference'), validate='one_to_one')
            assert len(aligned) == len(h) == len(expected)
            diff = float(np.max(abs(aligned.raw_prediction_actual - aligned.raw_prediction_reference)))
            assert diff < 1e-06
            rows.append(dict(dataset='Oxford', shots=shot, seed=seed, maximum_prediction_difference=diff))
            parts.append(h.sort_values(KEY).reset_index(drop=True))
        ensemble = parts[0].copy()
        ensemble['raw_prediction'] = np.mean([f.raw_prediction.to_numpy() for f in parts], axis=0)
        ensemble['seed'], ensemble['ensemble_components'] = (0, 3)
        oxford.append(ensemble)
    pd.concat(oxford, ignore_index=True).to_csv(folder / 'Oxford.csv.gz', index=False)
    write_report(out, 'external', rows)
    print('NASA and Oxford saved-model replay passed', flush=True)

def ablations(out):
    mod = module(ROOT / 'models/interventions.py', 'pipeline_interventions')
    mod.main()
    rows = []
    for arm in mod.ARMS:
        f = read(mod.OUT / arm / 'predictions_ensemble.csv')
        diff = compare(f, read(ROOT / f'results/reference/predictions/ablations/{arm}/predictions_ensemble.csv'), 1e-06)
        rows.append(dict(arm=arm, maximum_prediction_difference=diff))
    write_report(out, 'ablations', rows)
