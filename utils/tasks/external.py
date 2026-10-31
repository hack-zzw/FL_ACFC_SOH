"""Replay external models, or refit their locked support-only adapters."""
import argparse, json
from utils.cli import *
parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--dataset', choices=['NASA', 'Oxford'], default='Oxford')
parser.add_argument('--budget', type=int, default=20)
parser.add_argument('--seed', type=int, default=20260903)
parser.add_argument('--output')
parser.add_argument('--isolation-check', action='store_true')
args = parser.parse_args()
out = output_path(args.output or str(ROOT / 'results/generated/external_fit'))
if args.isolation_check:
    isolation_guard()
if not args.retrain:
    from utils.replays import external
    external(out)
    raise SystemExit(0)
import numpy as np, pandas as pd, torch
from types import SimpleNamespace
from models.external.source import read_config, load_scaler
from models.external.adaptation import input_features
from utils.statistics import read
torch.set_num_threads(4)
cfg = read_config(ROOT / 'configs/best_no_pinn_experiment_config.json')
device = torch.device('cuda')
if args.dataset == 'NASA':
    if args.budget not in [20, 40, 60]:
        parser.error('NASA budgets:20,40,60')
    from models.external.nasa_pipeline import clients_for, get_models
    from models.external.nasa_budgets import split_scale
    from models.external.nasa_prediction import option_predict
    clients = clients_for(SimpleNamespace(manifests=ROOT / 'data/external/manifests'))
    support, _, _, _ = split_scale(clients, args.budget)
    protocol = json.loads((ROOT / 'configs/external/NASA_external_protocol.json').read_text(encoding='utf-8'))
    recipe = protocol['candidates'][str(args.budget)]
    option = protocol['support_residual_policies'][str(args.budget)]
    folder = out / 'NASA'
    folder.mkdir(exist_ok=True)
    (folder / 'registration.json').write_text(json.dumps(dict(source='checkpoints/source', budget=args.budget, seed=args.seed, recipe=recipe)), encoding='utf-8')
    context = SimpleNamespace(output=folder, source=ROOT / 'checkpoints/source', cfg=cfg, device=device, scaler=load_scaler(ROOT / 'checkpoints/source/source_scaler.json'))
    models = get_models(context, 'registered', 'source_federated', args.budget, args.seed, recipe, support, args.budget)
    expected = read(ROOT / 'results/reference/external/NASA/personalization_predictions.csv')
    expected = expected[expected.option.eq('selected') & expected.shots.eq(args.budget) & expected.seed.eq(args.seed) & expected.query_mode.eq('matched_long_60to79')]
    errors = []
    predictions = []
    for client, g in expected.groupby('client', sort=True):
        full = clients[client].sort_values(['battery_id', 'cycle_index']).copy()
        full['record_index'] = full.groupby('battery_id').cumcount()
        q = g[['battery_id', 'record_index']].merge(full, on=['battery_id', 'record_index'], validate='one_to_one')
        pred, _ = option_predict(models['shared'], support[client], input_features(q), {k: recipe[k] for k in ['strength', 'alpha']}, option, device)
        errors.append(float(np.max(abs(pred - g.prediction.to_numpy()))))
        g = g.copy()
        g['prediction'] = pred
        predictions.append(g)
    frame = pd.concat(predictions, ignore_index=True)
else:
    if args.budget not in [0, 5, 10, 20]:
        parser.error('Oxford budgets:0,5,10,20')
    from dataloader.external import 读取外部数据清单
    from models.external.initialization import split_client, initialized_model
    from models.external.oxford import fit_predict
    from models.external.adaptation import infer
    folder = out / 'Oxford'
    folder.mkdir(exist_ok=True)
    full = 读取外部数据清单(ROOT / 'data/external/manifests/Oxford_10.json').sort_values(['battery_id', 'cycle_index'])
    protocol = json.loads((ROOT / 'configs/external/Oxford_external_protocol.json').read_text(encoding='utf-8'))
    evaluation_cells = {f'Cell{i}' for i in protocol['eval_cells']}
    full = full[full.battery_id.str.rsplit('_', n=1).str[-1].isin(evaluation_cells)].copy().reset_index(drop=True)
    assert full.battery_id.nunique() == 3
    support, query = split_client(full, args.budget)
    payload = torch.load(ROOT / f'checkpoints/source/source_seed_{args.seed}.pt', map_location='cpu', weights_only=True)
    scaler = load_scaler(ROOT / 'checkpoints/source/source_scaler.json')
    if args.budget == 0:
        model = initialized_model(cfg, payload['global_state'], device, args.seed)
        pred, _ = infer(model, scaler.变换(input_features(query)), device)
        state = {'candidate': {'method': 'zero_shot', 'transform': 'raw'}}
    else:
        policy = json.loads((ROOT / 'results/reference/external/Oxford/dataset_policies.json').read_text(encoding='utf-8'))
        candidate = policy['datasets']['Oxford'][str(args.budget)]['dataset_specific_acfc']
        pred, state, model = fit_predict(payload, cfg, support, input_features(query), scaler, device, candidate)
    frame = query[['battery_id', 'record_index', 'cycle_index']].copy()
    frame['raw_prediction'] = pred
    frame = frame[frame.record_index.ge(20)]
    expected = read(ROOT / 'results/reference/external/Oxford/evaluation_predictions.csv')
    expected = expected[expected.arm.eq('zero_shot' if args.budget == 0 else 'dataset_specific_acfc') & expected.shots.eq(20 if args.budget == 0 else args.budget) & expected.seed.eq(args.seed) & expected.record_index.ge(20)]
    merged = frame.merge(expected[['battery_id', 'record_index', 'raw_prediction']], on=['battery_id', 'record_index'], suffixes=('_actual', '_reference'), validate='one_to_one')
    errors = [float(np.max(abs(merged.raw_prediction_actual - merged.raw_prediction_reference)))]
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, adaptation=state), folder / f'adapter_{args.budget}_{args.seed}.pt')
maximum = max(errors)
assert maximum < 1e-06, maximum
frame.to_csv(folder / f'predictions_{args.budget}_{args.seed}.csv.gz', index=False)
report = dict(passed=True, dataset=args.dataset, budget=args.budget, seed=args.seed, query_records=len(frame), maximum_prediction_difference=maximum, query_labels_used_for_fitting=False)
(folder / f'verification_{args.budget}_{args.seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
